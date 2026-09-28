"""
lib/backbones — 骨架模型包: PointNet++, PointNet, PointMLP, PointNeXt.
"""

from lib.backbones._base import (
    SetAbstraction, FeaturePropagation,
    farthest_point_sample, ball_query, _index_points, _square_distance,
    FocalLoss, PerSampleWeightedCE, count_params,
)
from lib.backbones.pointnet2 import PointNetPlusPlus, GatedPointNetPlusPlus
from lib.backbones.pointnet import PointNetSeg
from lib.backbones.pointmlp import PointMLPSeg
from lib.backbones.pointnext import PointNeXtSeg

__all__ = [
    # 基础构建块
    "SetAbstraction", "FeaturePropagation",
    "farthest_point_sample", "ball_query", "_index_points", "_square_distance",
    "FocalLoss", "PerSampleWeightedCE", "count_params",
    # 骨架模型
    "PointNetPlusPlus", "GatedPointNetPlusPlus",
    "PointNetSeg", "PointMLPSeg", "PointNeXtSeg",
]
