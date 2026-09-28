"""几何基线包: 经典 (非学习) 点云分割方法, 用于与 PC-CascadeNet 对照."""

from .geometric_seg import (
    GeometricParams,
    LABEL_CONCRETE,
    LABEL_NOISE,
    LABEL_REBAR,
    LABEL_SLEEVE,
    compute_iou_np,
    local_pca_features,
    rs_iou_np,
    segment_geometric,
    statistical_outlier_mask,
)

__all__ = [
    "GeometricParams",
    "segment_geometric",
    "local_pca_features",
    "statistical_outlier_mask",
    "compute_iou_np",
    "rs_iou_np",
    "LABEL_NOISE",
    "LABEL_CONCRETE",
    "LABEL_REBAR",
    "LABEL_SLEEVE",
]
