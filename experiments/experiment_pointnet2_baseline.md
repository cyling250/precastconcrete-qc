# Experiment: PointNet++ Baseline

## 实验概述

使用 PointNet++ 对预制混凝土构件点云进行语义分割，将点云逐点分类为 **噪声（背景）**、**混凝土**、**钢筋** 三类。采用局部块（patch）采样的训练策略，通过 15cm 球查询 + KNN 补充从原始点云文件中抽取 8192 点子集，并配合钢筋比例筛选保证训练集中各类别的平衡。同时引入 40mm 邻域 PCA 几何特征（线性度/平面度/球度）作为额外输入通道。

## 配置

| 参数              |                  值                   |
| ----------------- | :-----------------------------------: |
| 输入点数          |                 8192                  |
| Batch size        |                   8                   |
| Epochs            |                  80                   |
| Learning rate     |        5e-4 (AdamW + CosineAnnealingLR)        |
| Weight decay      |                 1e-4                  |
| Gradient clipping |                  1.0                  |
| AMP (混合精度)    |                  ✅                   |
| 损失函数          | PerSampleWeightedCE (max_weight=15.0) |
| 优化器            |                 AdamW                 |
| 设备              |                 CUDA                  |
| 模型参数          |               502,915                 |
| 验证集模型选择标准 |              mIoU (最高)              |

## 数据

### 数据流水线架构

采用**双缓冲生产者-消费者架构**实现训练与数据生成的流水线并行：

```
主进程 (训练):
  Epoch 1: wait(ready) → 消费 Buffer A → ack → ... → wait(ready) → 消费 Buffer B

生产者进程 (训练):
  Epoch 1: 加载文件 → 构建 KDTree → 生成 patches → 写入共享内存 Buffer A → ready
        ← ack → 生成 Buffer B (与 Epoch 1 训练并行)
        ← ack → 生成 Buffer A (与 Epoch 2 训练并行)
        ...

生产者进程 (验证):
  独立进程，与训练生产者并行运行，同样使用双缓冲机制
```

同一时刻系统中最多存在 2 份完整数据：1 份正在被训练消费，1 份正在被生产者准备。通过 `multiprocessing.shared_memory` 实现零拷贝数据传输。

### 训练集 (augmented/)

| 项目 | 值 |
|------|-----|
| 路径 | `mock/res/point/augmented/` |
| 文件数 | 630 个 NPZ + 630 个 `_geo.npy` |
| 总点数 | ~1.6 亿点 |

### 数据生成

数据由 `mock/generate_synthetic_dataset.py` 自动生成，每类构件的样式参数均为**随机组合**：

| 构件类型 | 随机参数 | 范围 |
|---------|---------|------|
| 矩形柱 | 宽度 w, 深度 d, 高度 h, 钢筋直径 rd, 钢筋数量 rc | w,d∈[400,800]mm, h∈[2000,6000]mm, rd∈[16,32]mm, rc∈[4,12] |
| L 形柱 | w, d, h, 翼缘宽 wf, 腹板深 dw, rd, rc | w,d∈[500,900]mm, h∈[2000,6000]mm, rc∈[4,10] |
| 矩形梁 | 宽 b, 高 h, 长度 length, rd, rc | b∈[300,600]mm, h∈[400,900]mm, length∈[3000,8000]mm, rc∈[4,12] |
| T 形梁 | 翼缘宽 B, 腹板宽 bw, 高 h, 翼缘高 hf, length, rd, rc | B∈[600,1200]mm, rc∈[4,10] |
| 矩形叠合梁 | b, h, length, rd, rc, 栓钉直径 hd, 栓钉间距 hs | hd∈[8,12]mm, hs∈[200,500]mm |
| T 形叠合梁 | B, bw, h, hf, length, rd, rc, hd, hs | — |
| L 形叠合梁 | w, d, length, wf, dw, rd, rc, hd, hs | — |

### 循环结构

训练集共 **3 层嵌套循环**，总计 **630 个文件**：

```
7 种构件类型 × 10 种样式变体 (随机尺寸/钢筋参数)
           × 3 种点云密度 (high/medium/low: 0.02/0.01/0.005 pts/mm²)
           × 3 种全面噪声等级 (low/medium/high)
```

### 增强策略: 全面噪声等级

每种噪声等级一次性叠加全部 4 种噪声，不同等级参数不同：

| 噪声等级 | 块遮挡 (Occlusion) | 高斯抖动 (σ) | 随机丢弃 (Dropout) | 外来噪点 (Outlier) |
|---------|-------------------|-------------|------------------|-------------------|
| Low | 20% 表面遮挡 | σ=2mm | 保留 85% | 1% |
| Medium | 30% 表面遮挡 | σ=5mm | 保留 75% | 3% |
| High | 40% 表面遮挡 | σ=8mm | 保留 65% | 5% |

每等级均在点云空间随机生成遮挡块（模拟堆叠遮挡）、对点坐标施加高斯抖动（模拟扫描噪声）、随机丢弃部分点（模拟遮挡导致的点缺失）、以及添加外来噪点（模拟环境噪声）。

### 验证集 (val/)

| 项目 | 值 |
|------|-----|
| 路径 | `mock/res/point/val/` |
| 文件数 | 105 个 NPZ + 105 个 `_geo.npy` |
| 噪声策略 | 综合 18% 噪声 (1%+2%+5%+10% 一次性叠加) |
| 验证集与训练集关系 | **完全独立**，使用不同随机种子生成，钢筋参数/密度/噪声与训练集无重叠 |

验证集循环结构：
```
7 种构件类型 × 5 种样式变体 × 3 种点云密度
全部配以 18% 综合噪声（0.01+0.02+0.05+0.10=0.18 比例的外来噪点）
```

### 预计算几何特征 (40mm 邻域 PCA)

每个 NPZ 文件配对一个 `_geo.npy` 文件，包含 40mm 半径球邻域的 PCA 分析结果（3 通道，逐点）：

1. **线性度 (Linearity)**: (λ₂ − λ₁) / λ₂ — 反映点是否沿直线分布
2. **平面度 (Planarity)**: (λ₁ − λ₀) / λ₂ — 反映点是否在平面上
3. **球度 (Sphericity)**: λ₀ / λ₂ — 反映点是否均匀散布

其中 λ₀ ≤ λ₁ ≤ λ₂ 为邻域协方差矩阵的特征值。使用 `scipy.spatial.cKDTree` 查询 40mm 半径内的邻域点，最少要求 12 个点，不足时几何特征置零。

预计算脚本: `scripts/precompute_geo_features.py`，使用 `multiprocessing.Pool`（8 workers）并行处理。

## PointNet++ 模型架构

```
输入: (B, N, 6)
  └── XYZ (3通道, 归一化到单位球)
  └── 几何特征 (3通道: 线性度/平面度/球度)
                                ↓
┌──────────────────────────────────────────────────────────┐
│  SA1 — 第一级 Set Abstraction                            │
│    最远点采样 (FPS): 8192 → 512                          │
│    球查询 (Ball Query): radius=0.2 (归一化), nsample=32  │
│    物理半径: 80mm (scale 映射)                           │
│    MLP: 6→32 → 32→32 → 32→64                            │
│    输出: 512 × 64 (局部特征)                             │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  SA2 — 第二级 Set Abstraction                            │
│    最远点采样: 512 → 128                                  │
│    球查询: radius=0.4, nsample=64                         │
│    物理半径: 160mm                                       │
│    MLP: (64+3)→64 → 64→64 → 64→128                      │
│    输出: 128 × 128 (局部特征)                             │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  SA3 — 全局 Set Abstraction                              │
│    最远点采样: 128 → 1 (全局聚合)                        │
│    无球查询 (全局点)                                      │
│    MLP: (128+3)→128 → 128→256 → 256→512                  │
│    输出: 1 × 512 (全局特征向量)                           │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  FP3 — 特征传播 1                                       │
│    插值方式: 距离加权 (inverse distance weighting)       │
│    输入: SA2 × 128 channel + SA3 × 512 channel           │
│    跳跃连接: SA2 特征 (128维) + SA3 上采样 (512维)      │
│    MLP: 640→256 → 256→256                                │
│    输出: 128 × 256                                       │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  FP2 — 特征传播 2                                       │
│    输入: SA1 × 64 channel + FP3 × 256 channel            │
│    跳跃连接: SA1 特征 (64维) + FP3 上采样 (256维)       │
│    MLP: 320→128 → 128→128                                │
│    输出: 512 × 128                                       │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  FP1 — 特征传播 3 (到原始点数)                           │
│    输入: 输入点 × 6 channel + FP2 × 128 channel          │
│    跳跃连接: 无 (从输入点做 3 维插值意义不大)            │
│    MLP: 128→64 → 64→32                                   │
│    输出: 8192 × 32                                       │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  分类头 (Classification Head)                            │
│    Conv1D: 32→32, BatchNorm, ReLU                        │
│    Dropout(0.3) — 防止过拟合                             │
│    Conv1D: 32→3 (噪声/混凝土/钢筋)                       │
│    输出: (B, N, 3) — 逐点 logits                         │
└──────────────────────────────────────────────────────────┘
```

**模型参数**: 502,915
**输入维度**: 6 (XYZ + 3 几何特征)
**输出维度**: 3 类

### 动态物理半径机制

由于输入点云已经被归一化到单位球（`points /= scale`），Set Abstraction 层的球查询半径也需要相应缩放：

```python
r1 = 80.0 / scale   # SA1: 物理 80mm → 归一化半径
r2 = 160.0 / scale  # SA2: 物理 160mm → 归一化半径
```

其中 `scale` 是归一化时记录的物理尺度（单位为 mm），shape 为 `(B,)`，由数据集在 patch 采样时记录。

### 几何特征

几何特征 `geo_features` 为 `(B, N, 3)` 的张量，在 `model.forward()` 中与归一化 XYZ 坐标拼接为 `(B, N, 6)`，作为 SA1 的输入。

## 损失函数: `PerSampleWeightedCE`

逐样本动态加权交叉熵损失，对 batch 内的**每个样本独立计算**类别权重：

```
w_c = N / (3 × n_c),   clamped to [1.0, 15.0]
```

- N = 8192（该样本总点数）
- n_c = 该样本中第 c 类的点数
- 目标：使各类对 loss 的贡献大致相等，不受各类别点数比例影响
- 权重上限 15 倍：防止极端不平衡样本（如钢筋占 0.1%）的损失主导

### 设计动机

在钢筋点云比例极低（通常占总点数的 1%-10%）的情况下，标准的 CrossEntropyLoss 会使模型偏向于学习点数占比最大的混凝土类。PerSampleWeightedCE 通过逐样本动态加权，在每批次中独立提升少数类的 loss 贡献，避免全局权重策略中出现的"某一样本钢筋极少但权重溢出"问题。

### 归一化

每个样本的加权 loss 除以 N 进行归一化，使得 loss 的量级与标准交叉熵一致，便于学习率的调参。

## 训练集采样策略: PatchBasedDataset

### 数据预处理（一次性加载）

1. 加载 `augmented/` 目录的 630 个 NPZ 文件
2. 每个文件加载 point (N, 3)、label (N,) 及对应的 `_geo.npy` (N, 3)
3. 为每个文件构建 `scipy.spatial.cKDTree`（用于后续邻域查询）
4. 所有文件缓存到 `PersistCache` 对象中

### 每 Epoch Patch 生成 (`regenerate()`)

对每个文件按以下流程提取 8192 点子集：

```
1. 初始化可用掩码 available = True (所有点可用)
2. 从可用点中随机选一个 seed 点
3. 以 seed 为中心做 150mm 球查询 (PATCH_RADIUS_MM=150.0)
   a) 球内可用点 ≥ 8192 → 随机降采样至 8192
   b) 球内可用点 < 8192 → KNN 补充（从 seed 出发, 逐步扩大范围搜索可用点）
      - 使用 cKDTree.query(seed, k=16384) 获取近邻
      - 从未使用的近邻中逐点补充至 8192
   c) 仍不足 8192 → 跳过, 重新选种子 (连续失败 50 次 → 放弃该文件)
4. 标记选中的点为 unavailable (不重复使用)
5. 归一化:
   center = points - mean  # 去中心化
   scale = max(‖points‖)    # 物理尺度
   points = points / scale  # 归一化到单位球
6. 记录钢筋比例 rebar_ratio = (labels == 2).sum() / 8192
7. 重复步骤 2-6 直到当前文件可用点耗尽
```

### 筛选逻辑

所有 patches 按钢筋比例分组后筛选：

```
全部 patches → 打乱 →
  has_rebar (rebar_ratio > 0):  全部保留
  no_rebar  (rebar_ratio == 0):  保留 5 个 (KEEP_ZERO=5)
→ 合并 → 打乱 → 当前 epoch 训练数据
```

保留 5 个纯混凝土 patch 的目的是让模型见过"完全没有钢筋"的情况，避免验证时对纯混凝土区域产生误判。对钢筋比例接近 0% 的极端情况额外保留 5 个。

### 验证集采样策略

验证集采用**与训练集相同的局部邻域采样策略**（150mm ball query + KNN 补充），差异之处：

| 项目 | 训练 | 验证 |
|------|-----|------|
| 文件来源 | `augmented/` (630 文件) | `val/` (105 文件) |
| 筛选 | 保留 40% (按钢筋比例) | **不做筛选，全部保留** |
| 每 epoch | 重新生成 | 一次性生成**固定不变** |
| 生产者 | 独立进程 | 独立进程 |

不使用全局随机采样的原因：验证集需要与训练集一致的 patch 分布，避免因采样策略不同导致的"训练集 loss 无意义下降"问题。

## 数据加载: 生产者-消费者架构

### 共享内存缓冲区 `SharedPatchBuffer`

通过 `multiprocessing.shared_memory` 实现训练进程与数据生成进程之间的零拷贝数据传输：

| 缓冲区字段 | Shape | dtype | 说明 |
|-----------|-------|-------|------|
| `pts` | (3000, 8192, 3) | float32 | 归一化点坐标 |
| `geo` | (3000, 8192, 3) | float32 | 几何特征 (线性度/平面度/球度) |
| `lbl` | (3000, 8192) | int64 | 逐点标签 (0/1/2) |
| `scale` | (3000,) | float32 | 物理尺度 (mm) |
| `n` | (1,) | int64 | 有效 patch 数量 |

缓冲区最大容量 3000 patches。使用 `SharedMemory` 命名共享内存段，通过 `multiprocessing.Event` 同步读写。

### 生产-消费时序

```
生产者进程:          消费者进程 (训练/验证主进程):
  │                      │
  ├─ 创建共享内存          │
  ├─ 加载+KDTree          │
  ├─ regenerate()         │
  ├─ 写入 shm              │
  ├─ set(data_ready) ──── │
  │                      ├─ wait(data_ready)
  │                      ├─ 读取 n
  │                      ├─ set(ack) ──────────── │
  ├─ wait(ack) ◄────────  │
  ├─ regenerate() (并行)   ├─ 训练 (消费当前批次)
  ├─ 写入 shm              │
  ├─ set(data_ready) ──── │
  │                      ├─ wait(data_ready)
  │                      ├─ 读取 n
  │                      ├─ set(ack) ──────────── │
  │                      └─ 训练下一轮
```

训练和验证各有一个独立的生产者进程（均为 daemon=True），通过不同的 `suffix` 参数使用隔离的共享内存段。

### 双缓冲设计

训练进程和验证进程在各自 epoch 内的时序是**完全流水线化**的：

```
Epoch k+1 数据准备 (生产者)  ████████░░░░░░░░░░░░░░░
Epoch k 训练 (消费者)         ░░░░░░████████░░░░░░░░░░
```

同一时刻系统中最多存在 2 份完整数据：1 份正在被训练消费，1 份正在被生产者准备。通过这种方式，训练的计算等待时间被数据生成时间掩盖。

## 训练流程

```
每 epoch:
  1. make_train_loader()  ← 获取当前 epoch 的数据 (可能包含等待新数据)
  2. train_one_epoch()    ← 训练前向+反向传播
  3. make_val_loader()    ← 获取验证数据
  4. validate()           ← 验证: 计算 loss + 各类指标
  5. scheduler.step()     ← CosineAnnealingLR 学习率衰减
  6. 若 mIoU 提升 → 保存最佳模型到 checkpoints/best_pointnet2.pth
```

### 训练循环详细

```python
for pts, feat, lbl, scale in train_loader:
    pts, feat, lbl, scale = pts.to(DEVICE), feat.to(DEVICE), lbl.to(DEVICE), scale.to(DEVICE)
    
    with torch.amp.autocast("cuda"):       # 混合精度
        logits = model(pts, geo_features=feat, scale=scale)
        loss = criterion(logits, lbl)       # PerSampleWeightedCE
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model, max_norm=1.0)   # 梯度裁剪
    scaler.step(optimizer)
    scaler.update()
```

NaN/Inf 防护：若 loss 为 NaN 或 Inf，跳过该 batch 的梯度更新，继续下一 batch。

### 模型保存策略

使用 **mIoU**（而非 validation accuracy）作为模型保存标准。原因：
- mIoU 对各类别一视同仁，不受类别不平衡影响
- 噪声类的准确率可能误导整体准确率，而 mIoU 更严格

## 评估指标

| 指标 | 说明 |
|------|------|
| **噪声 IoU** | 背景噪声类的 IoU = TP/(TP+FP+FN) |
| **混凝土 IoU** | 混凝土类的 IoU |
| **钢筋 IoU** | 钢筋类的 IoU |
| **mIoU** | 三类 IoU 的均值 |
| **总体 Acc** | 所有点的正确率 |
| **P (Precision)** | 精确率 = TP/(TP+FP), 逐类计算 |
| **R (Recall)** | 召回率 = TP/(TP+FN), 逐类计算 |
| **F1** | F1 = 2×P×R/(P+R), 逐类计算 |

## 文件结构

```
experiments/
  experiment_pointnet2_baseline.py   ← 训练入口
  experiment_pointnet2_baseline.md   ← 本文档
  checkpoints/
    best_pointnet2.pth               ← 最佳模型权重 (mIoU 最高)
    history.npz                      ← 训练历史 (epoch 级指标)
    history.json                     ← 同上, JSON 格式
    best_metrics.json                ← 最佳模型指标
lib/pointnet2/
  __init__.py                        ← 统一导出
  model.py                           ← PointNet++ (SA/FP)+PerSampleWeightedCE
  dataset.py                         ← PatchBasedDataset+双缓冲生产者-消费者
  metrics.py                         ← 评估指标 (IoU, PRF, Acc)
mock/
  generate_synthetic_dataset.py      ← 合成数据生成器
  preview_mock.py                    ← 点云可视化工具
  res/point/
    original/                        ← 干净数据 (存档, 不用于训练)
    augmented/                       ← 增强数据 (训练集: 630文件+_geo.npy)
    val/                             ← 独立验证集 (105文件+_geo.npy)
scripts/
  precompute_geo_features.py         ← 预计算 40mm 几何特征
checkpoints/                         ← 模型训练输出目录 (自动创建)
```
