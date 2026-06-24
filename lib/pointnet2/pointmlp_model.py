"""
PointMLP: Rethinking Network Design and Local Geometry in Point Cloud.

Ma et al., ICLR 2022.

简化的分割版本: 保留核心的 Geometric Affine + Residual MLP 模块.
Forward interface: forward(xyz, geo_features=None, scale=None) -> (B, N, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .model import farthest_point_sample, _index_points, _square_distance


# ── Geometric Affine Module ──


class GeometricAffine(nn.Module):
    """
    PointMLP 的核心局部聚合: k-NN + 特征差 + 位置差 → MLP → 池化.

    对每个点取 k 个最近邻, 计算邻域特征偏移 (fj - fi) 和位置偏移 (pj - pi),
    拼接后通过 MLP 学习局部几何信息.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 24):
        super().__init__()
        self.k = k
        # 输入: 位置偏移 (pj - pi) + 特征偏移 (fj - fi) = 3 + in_ch
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch + 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, p: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (B, N, 3) points
            f: (B, C, N) features
        Returns:
            (B, C_out, N) aggregated features
        """
        B, C, N = f.shape
        p_t = p.transpose(-1, -2)  # (B, 3, N)

        # k-NN in 3D space
        sq_dist = _square_distance(p, p)  # (B, N, N)
        _, idx = sq_dist.topk(self.k + 1, dim=-1, largest=False)
        idx = idx[:, :, 1:]  # exclude self, (B, N, k)

        # Position differences (pj - pi)
        grouped_p = _index_points(p, idx)  # (B, N, k, 3)
        dp = grouped_p - p.unsqueeze(2)  # (B, N, k, 3)
        dp = dp.permute(0, 3, 1, 2)  # (B, 3, N, k)

        # Feature differences (fj - fi)
        grouped_f = _index_points(f.transpose(-1, -2), idx)  # (B, N, k, C)
        fj = grouped_f.permute(0, 3, 1, 2)  # (B, C, N, k)
        fi = f.unsqueeze(-1).expand(-1, -1, -1, self.k)  # (B, C, N, k)
        df = fj - fi  # (B, C, N, k)

        # Concatenate and apply MLP
        inp = torch.cat([dp, df], dim=1)  # (B, 3+C, N, k)
        out = self.mlp(inp)  # (B, C_out, N, k)
        out, _ = out.max(dim=-1)  # (B, C_out, N)
        return out


# ── Residual MLP Block ──


class ResMLPBlock(nn.Module):
    """
    残差 MLP 块: MLP(expand) → MLP(compress) + 残差连接.
    """

    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        mid_ch = int(channels * expansion)
        self.conv1 = nn.Conv1d(channels, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity, inplace=True)
        return x


# ── PointMLP Stages ──


class PointMLPStage(nn.Module):
    """
    PointMLP 一级: (可选降采样) → GeometricAffine → ResMLP.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 24,
        npoint: Optional[int] = None,
    ):
        super().__init__()
        self.npoint = npoint
        self.ga = GeometricAffine(in_ch, out_ch, k=k)
        self.res = ResMLPBlock(out_ch, expansion=2.0)
        # 跳跃连接: in_ch → out_ch (维度匹配)
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, p: torch.Tensor, f: torch.Tensor):
        """
        Returns:
            new_p: (B, N_new, 3) or same if no downsampling
            new_f: (B, C_out, N_new)
        """
        if self.npoint is not None and self.npoint < p.shape[1]:
            # FPS downsampling
            fps_idx = farthest_point_sample(p, self.npoint)  # (B, npoint)
            new_p = _index_points(p, fps_idx.unsqueeze(-1)).squeeze(2)
            # Gather features at sampled points
            new_f = _index_points(f.transpose(-1, -2), fps_idx.unsqueeze(-1)).squeeze(2)
            new_f = new_f.transpose(-1, -2)  # (B, C, npoint)
        else:
            new_p = p
            new_f = f

        # Geometric Affine
        f_ga = self.ga(new_p, new_f)  # (B, C_out, N_new)

        # Residual MLP (with skip for channel matching)
        f_skip = self.skip(new_f)
        f_out = self.res(f_ga + f_skip)
        return new_p, f_out


# ── PointMLP 分割模型 ──


class PointMLPSeg(nn.Module):
    """
    PointMLP 语义分割模型 (简化版, 保留核心设计).

    Forward: (B, N, 3) xyz, (B, N, 3) geo_features, (B,) scale
      → (B, N, num_classes) logits

    Args:
        num_classes: 输出类别数 (默认 3)
        in_channels: 输入特征维数 (默认 6 = XYZ + geo)
        widths: 各级宽度 (默认 [64, 128, 256])
        k: 局部聚合邻居数 (默认 24)
        nsample: 每级 FPS 降采样目标点数 (默认 [2048, 512, 128])
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 6,
        widths: List[int] = None,
        k: int = 24,
        nsample: List[int] = None,
    ):
        super().__init__()
        if widths is None:
            widths = [64, 128, 256]
        if nsample is None:
            nsample = [2048, 512, 128]

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Embedding: map input to first width
        self.embed = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], 1, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder stages
        encoder = []
        cur_ch = widths[0]
        for i, (w, np_pts) in enumerate(zip(widths, nsample)):
            encoder.append(PointMLPStage(cur_ch, w, k=k, npoint=np_pts))
            cur_ch = w
        self.encoder = nn.ModuleList(encoder)

        # Decoder: interpolation + skip + conv
        decoder = []
        dec_in = widths[-1]
        # skip channels: reversed(widths[:-1]) + [widths[0]] = [128, 64, 64]
        skip_chs = list(reversed(widths[:-1])) + [widths[0]]
        for i, skip_ch in enumerate(skip_chs):
            mlp_in = dec_in + skip_ch
            decoder.append(nn.Sequential(
                nn.Conv1d(mlp_in, skip_ch, 1, bias=False),
                nn.BatchNorm1d(skip_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(skip_ch, skip_ch, 1, bias=False),
                nn.BatchNorm1d(skip_ch),
                nn.ReLU(inplace=True),
            ))
            dec_in = skip_ch
        self.decoder = nn.ModuleList(decoder)

        # Head
        self.head = nn.Sequential(
            nn.Conv1d(dec_in, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, 1),
        )

    def _interpolate(self, p1, p2, f2):
        """3-NN 距离加权插值: p1 是目标 (密), p2 是源 (疏), f2 是源特征."""
        B, N1 = p1.shape[:2]
        N2 = p2.shape[1]
        if N2 == 1:
            return f2.expand(-1, -1, N1)
        sq = _square_distance(p1, p2)
        dist, idx = sq.topk(3, dim=-1, largest=False)
        w = 1.0 / (dist + 1e-8)
        w = w / w.sum(dim=-1, keepdim=True)
        feat_src = _index_points(f2.transpose(-1, -2), idx)
        interpolated = (feat_src * w.unsqueeze(-1)).sum(dim=2)
        return interpolated.transpose(-1, -2)

    def forward(
        self,
        xyz: torch.Tensor,
        geo_features: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            geo_features: (B, N, 3) geometric features
            scale: ignored (PointMLP 不使用动态半径, k-NN 在归一化空间固定)
        Returns:
            logits: (B, N, num_classes)
        """
        # Build input features
        if geo_features is not None:
            f = torch.cat([xyz, geo_features], dim=-1)  # (B, N, C_in)
        else:
            f = xyz
        f = f.transpose(-1, -2)  # (B, C_in, N)

        # Embedding
        f = self.embed(f)
        p = xyz

        # Storage for skip connections
        p_skips = [p]
        f_skips = [f]

        # Encoder
        for stage in self.encoder:
            p, f = stage(p, f)
            p_skips.append(p)
            f_skips.append(f)

        # Decoder
        dec_f = f_skips[-1]
        for i, dec_stage in enumerate(self.decoder):
            p_target = p_skips[-(i + 2)]
            f_skip = f_skips[-(i + 2)]
            p_source = p_skips[-(i + 1)]
            f_up = self._interpolate(p_target, p_source, dec_f)  # (B, C, N_target)
            cat_f = torch.cat([f_up, f_skip], dim=1)  # (B, C_in+C_skip, N_target)
            dec_f = dec_stage(cat_f)

        # Head
        logits = self.head(dec_f)  # (B, num_classes, N)
        return logits.transpose(-1, -2)  # (B, N, num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
