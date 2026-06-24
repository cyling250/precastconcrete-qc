"""基线: PointNet 基准网络."""

from _common import run_experiment, PointNetSeg


def main():
    run_experiment(
        model_factory=lambda: PointNetSeg(num_classes=4),
        tag="baseline_pointnet",
        desc="基线: PointNet",
    )


if __name__ == "__main__":
    main()
