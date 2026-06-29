"""SOTA: PointMLP + 标准CE + 物理感知采样 + 自适应球查询."""
import torch.nn as nn
from _common import run_experiment, PointMLPSeg


def main():
    run_experiment(
        model_factory=lambda: PointMLPSeg(num_classes=4),
        tag="baseline_pointmlp",
        desc="SOTA: PointMLP (ICLR 2022) + 标准CE + 物理感知采样",
        criterion=nn.CrossEntropyLoss(),
    )


if __name__ == "__main__":
    main()
