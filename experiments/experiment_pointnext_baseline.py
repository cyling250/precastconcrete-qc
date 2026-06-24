"""SOTA: PointNeXt 基准网络."""

from _common import run_experiment, PointNeXtSeg


def main():
    run_experiment(
        model_factory=lambda: PointNeXtSeg(num_classes=4),
        tag="baseline_pointnext",
        desc="SOTA: PointNeXt (NeurIPS 2022)",
    )


if __name__ == "__main__":
    main()
