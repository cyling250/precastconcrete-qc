"""主试验: PointNet++ + 标准CE + 物理感知采样 + 自适应球查询."""
import torch.nn as nn
from _common import run_experiment, PointNetPlusPlus


def main():
    run_experiment(
        model_factory=lambda: PointNetPlusPlus(num_classes=4, in_channels=3),
        tag="baseline_pointnet2",
        desc="主试验: PointNet++ + 标准CE + 物理感知采样 + 自适应球查询",
        batch_size=16,
        samples_per_epoch=3000,
        criterion=nn.CrossEntropyLoss(),
    )


if __name__ == "__main__":
    main()
