"""
实验共享配置: 架构注册表、默认超参数.
"""

# 支持的骨干架构
ARCH_REGISTRY = ["pointnet++", "pointnet", "pointmlp", "pointnext"]

# 架构 → 模型类映射
def _get_arch_map():
    from lib.backbones import (
        PointNetPlusPlus, PointNetSeg, PointMLPSeg, PointNeXtSeg,
    )
    return {
        "pointnet++": PointNetPlusPlus,
        "pointnet": PointNetSeg,
        "pointmlp": PointMLPSeg,
        "pointnext": PointNeXtSeg,
    }


def get_backbone_cls(arch: str):
    """根据架构名返回模型类."""
    return _get_arch_map()[arch]


# S1 默认配置
S1_DEFAULT_EPOCHS = 80
S1_DEFAULT_PATIENCE = 10

# S2 默认配置
S2_DEFAULT_EPOCHS = 80
S2_DEFAULT_VOXEL_MM = 40.0
S2_VAL_INTERVAL = 2

# A2 默认配置
A2_DEFAULT_EPOCHS = 120
