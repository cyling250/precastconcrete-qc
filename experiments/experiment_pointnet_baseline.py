"""
基准对比: PointNet 语义分割网络.

PointNet (Qi et al., CVPR 2017) — 简化版, 无 T-Net.

共享与主试验相同的数据流水线和损失函数 (PerSampleWeightedCE + 噪声感知),
仅更换骨干网络以对比不同架构的性能.
"""

from _common import run_experiment, PointNetSeg


def main():
    run_experiment(
        model_factory=lambda: PointNetSeg(num_classes=3),
        tag="baseline_pointnet",
        desc="基准: PointNet + 本文 loss + 数据预处理",
    )


if __name__ == "__main__":
    main()
