"""
SOTA 对比: PointMLP 语义分割网络.

PointMLP (Ma et al., ICLR 2022) — 纯 MLP 架构 + Geometric Affine 模块,
在多个基准上达到 SOTA 性能.

共享与主试验相同的数据流水线和损失函数 (PerSampleWeightedCE + 噪声感知),
仅更换骨干网络以对比 SOTA 架构的性能.
"""

from _common import run_experiment, PointMLPSeg


def main():
    run_experiment(
        model_factory=lambda: PointMLPSeg(num_classes=3),
        tag="baseline_pointmlp",
        desc="SOTA: PointMLP + 本文 loss + 数据预处理",
    )


if __name__ == "__main__":
    main()
