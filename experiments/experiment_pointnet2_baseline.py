"""主试验: PointNet++ 基准网络 + 本文提出的 loss 和数据预处理."""

from _common import run_experiment, PointNetPlusPlus


def main():
    run_experiment(
        model_factory=lambda: PointNetPlusPlus(num_classes=4),
        tag="baseline_pointnet2",
        desc="主试验: PointNet++ + 本文 loss + 数据预处理",
    )


if __name__ == "__main__":
    main()
