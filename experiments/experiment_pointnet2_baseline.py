"""
主试验: PointNet++ 基准网络 + 本文提出的 loss 和数据预处理.

通过更换 model_factory 可替换为其他骨干网络 (PointNet / PointNeXt / PointMLP 等),
共享相同的数据流水线和损失函数, 验证的是本文方法 (loss + 数据预处理) 的有效性.

架构:
  - 骨干: PointNet++ (三级 SA + 三级 FP, 502,915 参数)
  - 输入: XYZ + PCA 几何特征 (40mm 邻域, 线性度/平面度/球度)
  - 局部块: 150mm 球查询 + KNN 补充至 8192 点
  - 筛选: 含钢筋块全保留, 纯混凝土块每文件最多 5 个

损失函数 (方法论 2.4):
  - PerSampleWeightedCE: w_c = N / (3 × n_c), clamped [1.0, 15.0]
  - 噪声感知策略: α = 0.5 软降权

训练:
  - AdamW (lr=5e-4, weight_decay=1e-4), CosineAnnealingLR, 80 epochs
  - 梯度裁剪 max_norm=1.0
  - 双缓冲生产者-消费者 (共享内存零拷贝)
"""

from _common import run_experiment, PointNetPlusPlus


def main():
    run_experiment(
        model_factory=lambda: PointNetPlusPlus(num_classes=3),
        tag="baseline_pointnet2",
        desc="主试验: PointNet++ + 本文 loss + 数据预处理",
    )


if __name__ == "__main__":
    main()
