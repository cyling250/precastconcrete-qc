"""
消融 A1 — 删除自适应球查询: 使用标准 PointNet++ 固定半径 (0.2/0.4).

与主试验的唯一区别: 模型忽略 scale 参数, SA1/SA2 退化为固定归一化半径.
"""
import torch.nn as nn
from _common import (
    run_experiment, PointNetPlusPlus, NUM_CLASSES,
)


class FixedRadiusPointNetPlusPlus(PointNetPlusPlus):
    """PointNet++ with fixed radii — ignores scale for ablation."""
    def forward(self, xyz, geo_features=None, scale=None):
        return super().forward(xyz, geo_features=geo_features, scale=None)


def main():
    run_experiment(
        model_factory=lambda: FixedRadiusPointNetPlusPlus(num_classes=NUM_CLASSES, in_channels=3),
        tag="ablation_A1",
        desc="消融 A1 — 删除自适应球查询 (固定半径 0.2/0.4)",
        criterion=nn.CrossEntropyLoss(),
    )


if __name__ == "__main__":
    main()
