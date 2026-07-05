"""
PointNet2 分割模块 — 模型、数据集、评价指标、级联管线.

Sub-packages:
  - models:        PointNet++, PointNet, PointMLP, PointNeXt, losses
  - doublephase:   Cascade pipeline (S1 → adaptive voxel → S2)
  - dataset_utils: GPU-cached train loader, validation/test loaders
  - metrics_utils: IoU, precision/recall/F1, accuracy
"""

from .dataset_utils import (
    GpuCachedTrainLoader,
    RandomValDataset,
    RandomSampleDataset,
    create_dataloaders,
)
from .metrics_utils import (
    compute_iou,
    compute_precision_recall_f1,
    compute_accuracy,
)
from .models import (
    PointNetPlusPlus,
    GatedPointNetPlusPlus,
    PointNetSeg,
    PointMLPSeg,
    PointNeXtSeg,
    FocalLoss,
    PerSampleWeightedCE,
    count_params,
)
from .doublephase import (
    Stage2Model,
    filter_patch,
    build_s2_dataset,
    load_s2_loader,
    val_v2_simple,
    print_v2,
    S2_NAMES,
)
