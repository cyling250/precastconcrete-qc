"""
基准对比: PointNeXt 语义分割网络.

PointNeXt (Qian et al., NeurIPS 2022) — PointNet++ 改进版,
引入倒残差 MLP 和可扩展设计.

共享与主试验相同的数据流水线和损失函数 (PerSampleWeightedCE + 噪声感知),
仅更换骨干网络以对比不同架构的性能.
"""

from _common import run_experiment, PointNeXtSeg


def main():
    run_experiment(
        model_factory=lambda: PointNeXtSeg(num_classes=3),
        tag="baseline_pointnext",
        desc="基准: PointNeXt + 本文 loss + 数据预处理",
    )


if __name__ == "__main__":
    main()
