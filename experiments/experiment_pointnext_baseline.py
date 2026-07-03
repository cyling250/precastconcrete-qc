"""SOTA: PointNeXt + 标准CE + 物理感知采样 + 自适应球查询."""
import torch.nn as nn
from _common import run_experiment, PointNeXtSeg


def main():
    run_experiment(
        model_factory=lambda: PointNeXtSeg(num_classes=4),
        tag="baseline_pointnext",
        desc="SOTA: PointNeXt (NeurIPS 2022) + 标准CE + 物理感知采样",
        batch_size=16,
        samples_per_epoch=3000,
        criterion=nn.CrossEntropyLoss(),
    )


if __name__ == "__main__":
    main()
