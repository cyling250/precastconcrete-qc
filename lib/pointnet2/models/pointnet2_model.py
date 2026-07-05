"""
PointNet++ 分割模型 + Gated PointNet++ + Loss 函数.

架构:
  SA1: N→512, radius=0.2, MLP[32,32,64]
  SA2: 512→128, radius=0.4, MLP[64,64,128]
  SA3: 128→1 (global), MLP[128,256,512]
  FP3→FP2→FP1 上采样解码 + 跳跃连接
  Head: Conv1D → num_classes

支持动态物理半径: 传入 scale (mm) 将 80mm/160mm 物理半径映射到归一化空间.
参考: Qi et al., NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ._pointnet2_base import SetAbstraction, FeaturePropagation


# ── 主网络 ──


class PointNetPlusPlus(nn.Module):
    """(B, N, 3) → (B, N, num_classes).

    in_channels=6: 输入为 XYZ(3ch) + 几何特征(3ch).
    in_channels=3: 仅使用 XYZ, 忽略 geo_features.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.sa1 = SetAbstraction(512, 0.2, 32, in_channels, [32, 32, 64])
        self.sa2 = SetAbstraction(128, 0.4, 64, 64 + 3, [64, 64, 128])
        self.sa3 = SetAbstraction(1, None, None, 128 + 3, [128, 256, 512])
        self.fp3 = FeaturePropagation(128 + 512, [256, 256])
        self.fp2 = FeaturePropagation(64 + 256, [128, 128])
        self.fp1 = FeaturePropagation(128, [64, 32])
        self.head = nn.Sequential(
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(32, num_classes, 1),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        geo_features: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """scale: (B,) 归一化尺度 (mm). 若提供, SA1=80mm, SA2=160mm."""
        if scale is not None:
            scale = scale.to(xyz.device, non_blocking=True)
            r1 = 80.0 / scale
            r2 = 160.0 / scale
        else:
            r1 = r2 = None

        # in_channels=3 时忽略 geo_features（仅使用 XYZ）
        if self.in_channels > 3 and geo_features is not None:
            feat_for_sa1 = geo_features.transpose(-1, -2)
        else:
            feat_for_sa1 = None
        l1_xyz, l1_f = self.sa1(xyz, features=feat_for_sa1, radius_override=r1)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f, radius_override=r2)
        l3_xyz, l3_f = self.sa3(l2_xyz, l2_f)

        fp3 = self.fp3(l2_xyz, l3_xyz, l2_f, l3_f)
        fp2 = self.fp2(l1_xyz, l2_xyz, l1_f, fp3)
        fp1 = self.fp1(xyz, l1_xyz, None, fp2)

        logits = self.head(fp1)
        return logits.transpose(-1, -2)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Loss 函数 ──


class FocalLoss(nn.Module):
    """Focal Loss: 自动聚焦难分类样本, 压制已分对样本的 loss 贡献.

    FL = -α_c · (1-p_c)^γ · log(p_c)

    当 γ>0 时:
      - 已分对的点 (p_c → 1):  (1-p_c)^γ → 0, loss 被压制
      - 分错的点 (p_c → 0):  (1-p_c)^γ → 1, loss 保留
    α_c 可同时做类别加权.

    Args:
        alpha: 类别权重, 形状 (C,) 或标量. None 则等权.
        gamma: 聚焦参数 (默认 2.0)
        max_weight: 类别权重的上限 (防止过拟合)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        max_weight: float = 15.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.max_weight = max_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C, N), targets: (B, N)"""
        # 每个点的 CE loss
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # 全局类别加权
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device, non_blocking=True)
            # alpha 形状 (C,), 按 targets 展开
            weight_per_point = alpha[targets]  # (B, N)
            # 限制最大权重
            if self.max_weight > 0:
                weight_per_point = torch.clamp(weight_per_point, max=self.max_weight)
            ce_loss = ce_loss * weight_per_point

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class PerSampleWeightedCE(nn.Module):
    """逐样本动态加权交叉熵: 根据当前样本各类点数计算权重.

    公式 (完整版, 4 类全参与):
        w_c = noise_weight × N / (4 × n_c)   (c=0, 噪声降权)
        w_c = N / (4 × n_c)                   (c=1,2,3, 正常加权)
        均 clamped to ≤ max_weight
        最终 loss = Σ(w_c · CE) / N

    N = 样本总点数, n_c = 该类当前样本的点数
    噪声类以 noise_weight 比例降低 loss 贡献, 保留微弱梯度防止 logit 漂移.
    """

    def __init__(self, max_weight: float = 15.0, noise_weight: float = 0.5):
        super().__init__()
        self.max_weight = max_weight
        self.noise_weight = noise_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C, N), targets: (B, N)"""
        ce = F.cross_entropy(logits, targets, reduction="none")  # (B, N)
        B, N = targets.shape

        weight_map = torch.ones_like(ce)
        for b in range(B):
            for c in range(4):  # 0=噪声, 1=混凝土, 2=钢筋, 3=套筒
                n_c = (targets[b] == c).sum().item()
                if n_c == 0:
                    continue
                w = N / (4.0 * n_c)
                w = min(w, self.max_weight)
                if c == 0:  # 噪声降权
                    w *= self.noise_weight
                mask = targets[b] == c
                weight_map[b, mask] = w

        # 除以 N 归一化到标准 CE 量级
        loss = (ce * weight_map).sum(dim=1).mean() / N
        return loss


# ═══════════════════════════════════════════════════════════
#  Gated PointNet++: 门控多任务分割 (YOLO风格)
# ═══════════════════════════════════════════════════════════


class GatedPointNetPlusPlus(nn.Module):
    """门控 PointNet++ (YOLO风格).

    架构:
      共享 backbone → fp1 (32维/点)
        ├── coarse_head(3类): 噪声=0, 混凝土=1, 金属=2     ← 全点参与
        └── fine_head(2类):   钢筋=0, 套筒=1               ← 全点推理, 门控

    门控:
      训练: ignore_index=-1 自动忽略非金属点的 fine loss
      推理: coarse_pred==2 的位置由 fine_logits 覆盖

    端到端一次 forward, 无分阶段/逐样本/拼接.
    """

    def __init__(
        self, num_classes: int = 4, in_channels: int = 3, lambda_fine: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lambda_fine = lambda_fine

        # ── 共享编码器-解码器 ──
        self.sa1 = SetAbstraction(512, 0.2, 32, in_channels, [32, 32, 64])
        self.sa2 = SetAbstraction(128, 0.4, 64, 64 + 3, [64, 64, 128])
        self.sa3 = SetAbstraction(1, None, None, 128 + 3, [128, 256, 512])
        self.fp3 = FeaturePropagation(128 + 512, [256, 256])
        self.fp2 = FeaturePropagation(64 + 256, [128, 128])
        self.fp1 = FeaturePropagation(128, [64, 32])

        # ── 粗分割头: 3 类 (噪声/混凝土/金属) ──
        self.coarse_head = nn.Sequential(
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(32, 3, 1),
        )

        # ── 精细分割头: 2 类 (钢筋/套筒) ──
        self.fine_head = nn.Sequential(
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(32, 2, 1),
        )

    def _encode_decode(self, xyz, geo_features, scale):
        if scale is not None:
            scale = scale.to(xyz.device, non_blocking=True)
            r1 = 80.0 / scale
            r2 = 160.0 / scale
        else:
            r1 = r2 = None
        if self.in_channels > 3 and geo_features is not None:
            feat_for_sa1 = geo_features.transpose(-1, -2)
        else:
            feat_for_sa1 = None
        l1_xyz, l1_f = self.sa1(xyz, features=feat_for_sa1, radius_override=r1)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f, radius_override=r2)
        l3_xyz, l3_f = self.sa3(l2_xyz, l2_f)
        fp3 = self.fp3(l2_xyz, l3_xyz, l2_f, l3_f)
        fp2 = self.fp2(l1_xyz, l2_xyz, l1_f, fp3)
        fp1 = self.fp1(xyz, l1_xyz, None, fp2)
        return fp1

    def forward(self, xyz, geo_features=None, scale=None, labels=None):
        """一次 forward: fine_head 在全点 fp1 上推理, ignore_index 门控."""
        fp1 = self._encode_decode(xyz, geo_features, scale)  # (B, 32, N)
        coarse_logits = self.coarse_head(fp1)  # (B, 3, N)
        fine_logits = self.fine_head(fp1)  # (B, 2, N)

        if self.training and labels is not None:
            return self._forward_loss(coarse_logits, fine_logits, labels, xyz.device)
        return self._forward_eval(coarse_logits, fine_logits, xyz.device)

    def _forward_loss(self, coarse_logits, fine_logits, labels, device):
        B, N = labels.shape

        # ── Coarse loss: 3 类, 全点 ──
        coarse_labels = labels.clone()
        coarse_labels[coarse_labels == 3] = 2  # 套筒→金属
        coarse_loss = F.cross_entropy(coarse_logits, coarse_labels)

        # ── Fine loss: 2 类, ignore_index=-1 门控 ──
        fine_labels = torch.full_like(labels, -1)
        fine_labels[labels == 2] = 0  # 钢筋
        fine_labels[labels == 3] = 1  # 套筒
        fine_loss = F.cross_entropy(fine_logits, fine_labels, ignore_index=-1)

        total_loss = coarse_loss + self.lambda_fine * fine_loss

        # ── 合并 logits (教师强制门控) ──
        B4 = torch.zeros(B, 4, N, device=device)
        B4[:, 0, :] = coarse_logits[:, 0, :]  # 噪声
        B4[:, 1, :] = coarse_logits[:, 1, :]  # 混凝土
        B4[:, 2, :] = coarse_logits[:, 2, :]  # 钢筋(默认)
        B4[:, 3, :] = coarse_logits[:, 2, :]  # 套筒(默认)
        metal = (labels == 2) | (labels == 3)
        B4[:, 2, :][metal] = fine_logits[:, 0, :][metal]
        B4[:, 3, :][metal] = fine_logits[:, 1, :][metal]

        return B4.permute(0, 2, 1), total_loss, coarse_loss, fine_loss

    @torch.no_grad()
    def _forward_eval(self, coarse_logits, fine_logits, device):
        B, N = coarse_logits.shape[0], coarse_logits.shape[2]
        coarse_pred = coarse_logits.argmax(dim=1)  # (B, N)

        out = torch.zeros(B, N, 4, device=device)
        out[:, :, 0] = coarse_logits[:, 0, :]  # 噪声
        out[:, :, 1] = coarse_logits[:, 1, :]  # 混凝土
        out[:, :, 2] = coarse_logits[:, 2, :]  # 钢筋(默认)
        out[:, :, 3] = coarse_logits[:, 2, :]  # 套筒(默认)

        metal = coarse_pred == 2
        out[:, :, 2] = coarse_logits[:, 2, :]
        out[:, :, 3] = coarse_logits[:, 2, :]

        metal = (coarse_pred == 2)
        out[:, :, 2][metal] = fine_logits[:, 0, :][metal]
        out[:, :, 3][metal] = fine_logits[:, 1, :][metal]

        return out
