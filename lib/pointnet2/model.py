"""
PointNet++ 分割模型.

架构:
  SA1: N→512, radius=0.2, MLP[32,32,64]
  SA2: 512→128, radius=0.4, MLP[64,64,128]
  SA3: 128→1 (global), MLP[128,256,512]
  FP3→FP2→FP1 上采样解码 + 跳跃连接
  Head: Conv1D → 3 类 (噪声/混凝土/钢筋)

支持动态物理半径: 传入 scale (mm) 将 80mm/160mm 物理半径映射到归一化空间.
参考: Qi et al., NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── 工具函数 ──


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(B, S, k) → (B, S, k, C) 按索引取点."""
    B = points.shape[0]
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand_as(idx)
    return points[batch_indices, idx.long()]


def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """(B, N, C) x (B, M, C) → (B, N, M) 平方欧氏距离."""
    dot = torch.matmul(src, dst.transpose(-1, -2))
    src_sq = (src**2).sum(dim=-1, keepdim=True)
    dst_sq = (dst**2).sum(dim=-1, keepdim=True)
    return src_sq - 2 * dot + dst_sq.transpose(-1, -2)


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS: (B, N, 3) → (B, npoint)"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_pts = xyz[batch_indices, farthest].unsqueeze(1)
        dist = ((xyz - centroid_pts) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=-1)[1]
    return centroids


def ball_query(
    radius: float | torch.Tensor, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """球查询: (B, S, nsample) 索引."""
    sq_dist = _square_distance(new_xyz, xyz)  # (B, S, N)
    if isinstance(radius, torch.Tensor):
        radius = radius.view(-1, 1, 1)  # (B, 1, 1)
    mask = sq_dist > radius**2
    sq_dist[mask] = float("inf")
    _, idx = sq_dist.sort(dim=-1, descending=False)
    return idx[:, :, :nsample]


# ── Set Abstraction ──


class SetAbstraction(nn.Module):
    """SA: 采样 → 分组 → 归一化 → MLP → 最大池化."""

    def __init__(self, npoint: int, radius: float, nsample: int, in_ch: int, mlp: list):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        layers = []
        ch = in_ch
        for out_ch in mlp:
            layers += [
                nn.Conv2d(ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        radius_override: Optional[torch.Tensor] = None,
    ):
        if self.npoint is None:
            new_xyz = xyz
            idx = (
                torch.arange(xyz.shape[1], device=xyz.device)
                .view(1, 1, -1)
                .expand(xyz.shape[0], 1, -1)
            )
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = _index_points(xyz, fps_idx.unsqueeze(-1)).squeeze(2)

        if self.radius is None:
            grouped_xyz = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)
            if features is not None:
                grouped_f = features.unsqueeze(2)
                if new_xyz.shape[1] > 1:
                    grouped_f = grouped_f.expand(-1, -1, new_xyz.shape[1], -1)
        else:
            effective_radius = radius_override if radius_override is not None else self.radius
            idx = ball_query(effective_radius, self.nsample, xyz, new_xyz)
            grouped_xyz = _index_points(xyz, idx) - new_xyz.unsqueeze(2)
            grouped_f = (
                _index_points(features.transpose(-1, -2), idx).permute(0, 3, 1, 2)
                if features is not None
                else None
            )

        inp = grouped_xyz.permute(0, 3, 1, 2)
        if grouped_f is not None:
            inp = torch.cat([inp, grouped_f], dim=1)
        out = self.mlp(inp)
        out, _ = out.max(dim=-1)
        return new_xyz, out


# ── Feature Propagation ──


class FeaturePropagation(nn.Module):
    """FP: 距离加权插值 + 跳跃连接 + MLP."""

    def __init__(self, in_ch: int, mlp: list):
        super().__init__()
        layers = []
        ch = in_ch
        for out_ch in mlp:
            layers += [
                nn.Conv1d(ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        feat1: Optional[torch.Tensor],
        feat2: torch.Tensor,
    ) -> torch.Tensor:
        B, N1 = xyz1.shape[:2]
        N2 = xyz2.shape[1]
        if N2 == 1:
            interpolated = feat2.expand(-1, -1, N1)
        else:
            sq = _square_distance(xyz1, xyz2)
            dist, idx = sq.topk(3, dim=-1, largest=False)
            w = 1.0 / (dist + 1e-8)
            w = w / w.sum(dim=-1, keepdim=True)
            feat_src = _index_points(feat2.transpose(-1, -2), idx)
            interpolated = (feat_src * w.unsqueeze(-1)).sum(dim=2).transpose(-1, -2)
        cat_feat = interpolated if feat1 is None else torch.cat([interpolated, feat1], dim=1)
        return self.mlp(cat_feat)


# ── 主网络 ──


class PointNetPlusPlus(nn.Module):
    """(B, N, 3) → (B, N, num_classes)."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.sa1 = SetAbstraction(512, 0.2, 32, 6, [32, 32, 64])
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

    def forward(self, xyz: torch.Tensor, geo_features: Optional[torch.Tensor] = None,
                scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """scale: (B,) 归一化尺度 (mm). 若提供, SA1=80mm, SA2=160mm."""
        if scale is not None:
            scale = scale.to(xyz.device, non_blocking=True)
            r1 = 80.0 / scale
            r2 = 160.0 / scale
        else:
            r1 = r2 = None

        # geo_features: (B, N, 3) → (B, 3, N) 与 SA 层的 channel-first 格式对齐
        if geo_features is not None:
            geo_features = geo_features.transpose(-1, -2)
        l1_xyz, l1_f = self.sa1(xyz, features=geo_features, radius_override=r1)
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

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 max_weight: float = 15.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.max_weight = max_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C, N), targets: (B, N)"""
        # 每个点的 CE loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

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
        ce = F.cross_entropy(logits, targets, reduction='none')  # (B, N)
        B, N = targets.shape

        weight_map = torch.ones_like(ce)
        for b in range(B):
            for c in range(4):  # 0=噪声, 1=混凝土, 2=钢筋, 3=套筒
                n_c = (targets[b] == c).sum().item()
                if n_c == 0:
                    continue
                w = N / (4.0 * n_c)
                w = min(w, self.max_weight)
                if c == 0:          # 噪声降权
                    w *= self.noise_weight
                mask = (targets[b] == c)
                weight_map[b, mask] = w

        # 除以 N 归一化到标准 CE 量级
        loss = (ce * weight_map).sum(dim=1).mean() / N
        return loss
