"""
Shared PointNet++ building blocks: sampling, grouping, SA/FP layers.

These are the core operations reused by PointNet++, PointMLP, and PointNeXt.
"""

import torch
import torch.nn as nn
from typing import Optional


# ── Utility functions ──


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
            effective_radius = (
                radius_override if radius_override is not None else self.radius
            )
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
        cat_feat = (
            interpolated if feat1 is None else torch.cat([interpolated, feat1], dim=1)
        )
        return self.mlp(cat_feat)
