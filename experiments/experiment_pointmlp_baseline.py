"""SOTA: PointMLP 基准网络."""

from _common import run_experiment, PointMLPSeg


def main():
    run_experiment(
        model_factory=lambda: PointMLPSeg(num_classes=4),
        tag="baseline_pointmlp",
        desc="SOTA: PointMLP (ICLR 2022)",
    )


if __name__ == "__main__":
    main()
