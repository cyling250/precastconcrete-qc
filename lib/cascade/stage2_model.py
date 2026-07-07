"""
S2 模型: PointNet++ 6通道输入 (XYZ + S1概率), 输出 4 类.
"""

from lib.backbones.pointnet2 import PointNetPlusPlus


class Stage2Model(PointNetPlusPlus):
    """S2: 输入 XYZ(3) + S1_probs(3), 输出 (B,N,4)."""

    def __init__(self, in_channels: int = 6):
        super().__init__(num_classes=4, in_channels=in_channels)

    def forward(self, xyz, s1_probs, scale=None):
        """xyz: (B,N,3), s1_probs: (B,N,3)"""
        return super().forward(xyz, geo_features=s1_probs, scale=None)
