"""
消融 A4 — 无采样筛选 (keep_ratio=1.0, 随机采样, 不做钢筋筛选).
"""

from _common import (
    console, NUM_CLASSES, set_seed, run_experiment,
    PointNetPlusPlus, PerSampleWeightedCE,
)

TAG = "A4"
DESC = "A4 — 无采样筛选 (keep_ratio=1.0, 随机采样)"


def main():
    set_seed(42)
    run_experiment(
        model_factory=lambda: PointNetPlusPlus(num_classes=NUM_CLASSES),
        tag=f"ablation_{TAG}",
        desc=DESC,
        keep_ratio=1.0,
    )


if __name__ == "__main__":
    main()
