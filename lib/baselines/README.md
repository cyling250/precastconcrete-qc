# 几何基线（Geometric baseline）

经典（非学习）材料级点云分割基线，用于与 PC-CascadeNet 在同一合成测试集、同一指标口径下对照。
纯 CPU（numpy + scipy），不依赖 torch / CUDA。

## 方法

沿用稿件引言所引用的几何方法谱系：

1. **固定物理半径邻域的 PCA 特征值描述子** — `linearity / planarity / sphericity / curvature`
   （Kim et al. 2020, Autom. Constr. 112:103077；Wang et al. 2017, CACAIE 32:787）
2. **统计离群点剔除（SOR）** — 平均近邻距离超出 `median + α·1.4826·MAD` 判为噪声
3. **RANSAC 圆柱拟合** — 由法向叉积定轴向、投影圆拟合定轴位与半径（Fischler & Bolles 1981）
4. **自适应容差的 RANSAC 平面拟合** — 混凝土表面

## 实测能力边界（重要）

**四分类中几何方法给不出"套筒"类。** 全部 184 个配置（96 主表 + 88 噪声曲线）中套筒 IoU 恒为 0.000。
原因（已实测确认，见 `_dbg_sleeve_gt.py`）：

- 套筒可见点集含底部圆盘与端面环，用真值点拟合圆柱得到半径 75~100 mm（真实外半径 24~33 mm），径向残差 RMS 27~45 mm；
- 壁厚仅 6 mm，与对抗扫描抖动所需的径向容差同量级，内外壁无法用单一圆柱分开；
- 基于曲率的候选取被混凝土以约 23:1 压制，贪心 RANSAC 恒选中混凝土长棱边（轴向跨度约等于柱高）。

因此本基线默认工作在 **三分类口径（噪声 / 混凝土 / 金属）**，与主模型的 Stage 1 对齐（`three_class=True`）。
四分类口径（`--four-class`）仅作对照，套筒列必然为 0。

## 密度敏感性（同时是引言论点的实证）

kNN 邻域会随点云密度改变物理尺度，使 PCA 描述子依赖扫描分辨率：

| 邻域方式 | low | medium | high |
|---|---|---|---|
| 固定点数 kNN（k=24） | 0.597 | 0.306 | 0.116 |
| **固定物理半径（40 mm）** | 0.690 | 0.765 | 0.557 |

（rebar 配筋型式、clean 数据下的金属类 IoU）

## 用法

```bash
# 自检：用项目生成器造构件, 直接在真值上评估
python -m lib.baselines.selftest --types rectangular_column --variants rebar sleeve
python -m lib.baselines.selftest --four-class          # 四分类口径对照

# 主实验: 标准测试集指标 + 11 档噪声退化曲线
python -m experiments.experiment_b1_geometric_baseline --sweep --out reports/b1_geo

# 诊断脚本
python -m lib.baselines.diag_features          # 逐类特征分布 / 阈值标定
python -m lib.baselines._dbg_vram              # 各骨干峰值显存
python -m lib.baselines._dbg_dataset_stats     # 类别占比 / 点保留率 / 标签翻转率
python -m lib.baselines._dbg_summarize         # 汇总 b1 实验 JSON
python -m lib.baselines._dbg_sleeve_gt         # 真值套筒的可拟合性上界
python -m lib.baselines._dbg_radius            # 邻域半径 x 线性度阈值扫描
```

## 调参说明

`GeometricParams` 的默认值是**在本数据集上实测标定**的（`lin_min=0.80`、`feat_radius_mm=40`、
`sor_alpha=3.5`）。论文中报告几何基线时，阈值应在验证集上扫描后报最优，而不是直接采用默认值。
所有阈值来源见 `diag_features` 输出的逐类分位数。
