"""
消融 A3a — 噪声感知 alpha=0.0 (完全忽略噪声类).

PerSampleWeightedCE 中 noise_weight=0.0, 噪声点不计入 loss.
"""

from _common import (
    console, NUM_CLASSES, set_seed, run_experiment,
    PointNetPlusPlus, PerSampleWeightedCE,
)

TAG = "A3a"
DESC = "A3a — 噪声感知 alpha=0.0 (完全忽略噪声类)"


def main():
    set_seed(42)
    run_experiment(
        model_factory=lambda: PointNetPlusPlus(num_classes=NUM_CLASSES),
        tag=f"ablation_{TAG}",
        desc=DESC,
        criterion=PerSampleWeightedCE(max_weight=15.0, noise_weight=0.0),
    )


if __name__ == "__main__":
    main()
