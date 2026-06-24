"""
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.

Qi et al., CVPR 2017.

简化的分割版本: 无 T-Net, 保留核心的 per-point MLP + global max pool 结构.
Forward interface: forward(xyz, geo_features=None, scale=None) -> (B, N, C)
"""

import torch
import torch.nn as nn
from typing import Optional


class PointNetSeg(nn.Module):
    """
    PointNet 语义分割模型 (无 T-Net 简化版).

    (B, N, 3) xyz + (B, N, 3) geo_features → (B, N, num_classes)

    Args:
        num_classes: 输出类别数 (默认 3)
        in_channels: 输入特征维数 (默认 6 = 3 XYZ + 3 geo)
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Per-point MLP encoder
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # Decoder: global + local concat → per-point MLP
        self.decoder = nn.Sequential(
            nn.Conv1d(1024 + 64, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        geo_features: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            geo_features: (B, N, 3) geometric features (linearity, planarity, sphericity)
            scale: ignored (PointNet 不使用动态半径)
        Returns:
            logits: (B, N, num_classes)
        """
        # Build input features
        if geo_features is not None:
            f = torch.cat([xyz, geo_features], dim=-1)  # (B, N, 6)
        else:
            f = xyz  # (B, N, 3)
        f = f.transpose(-1, -2)  # (B, C, N)

        # Encoder
        local_feat = self.mlp1(f)  # (B, 64, N)
        global_feat = self.mlp2(local_feat)  # (B, 1024, N)
        global_feat_pooled = global_feat.max(dim=-1, keepdim=True)[0]  # (B, 1024, 1)

        # Concatenate global + local for per-point prediction
        global_expanded = global_feat_pooled.expand(-1, -1, local_feat.shape[-1])  # (B, 1024, N)
        cat_feat = torch.cat([global_expanded, local_feat], dim=1)  # (B, 1088, N)

        # Decoder + Head
        feat = self.decoder(cat_feat)  # (B, 128, N)
        logits = self.head(feat)  # (B, num_classes, N)
        return logits.transpose(-1, -2)  # (B, N, num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
