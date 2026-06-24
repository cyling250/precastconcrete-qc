"""
消融 A3b — 噪声感知 α=1.0 (平等对待噪声类).

PerSampleWeightedCE 中 noise_weight=1.0, 噪声与其他类同等加权.
"""

from _common import (
    console, NUM_CLASSES, set_seed, run_experiment,
    PointNetPlusPlus, PerSampleWeightedCE,
)

TAG = "A3b"
DESC = "A3b — 噪声感知 α=1.0 (平等对待噪声类)"


def main():
    set_seed(42)
    run_experiment(
        model_factory=lambda: PointNetPlusPlus(num_classes=NUM_CLASSES),
        tag=f"ablation_{TAG}",
        desc=DESC,
        criterion=PerSampleWeightedCE(max_weight=15.0, noise_weight=1.0),
    )


if __name__ == "__main__":
    main()
