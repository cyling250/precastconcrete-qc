"""基线: PointNet + 标准CE + 物理感知采样 + 自适应球查询."""
import torch.nn as nn
from _common import run_experiment, PointNetSeg


def main():
    run_experiment(
        model_factory=lambda: PointNetSeg(num_classes=4),
        tag="baseline_pointnet",
        desc="基线: PointNet + 标准CE + 物理感知采样 + 自适应球查询",
        criterion=nn.CrossEntropyLoss(),
    )


if __name__ == "__main__":
    main()
