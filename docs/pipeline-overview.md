# 点云分割流水线（Mock → 训练 → 评估）

> 本文档描述从合成点云数据就绪后，到模型训练完成并输出评估指标的完整流程。

---

## 1. 数据准备

### 1.1 目录结构

```
mock/inuse/
├── train/  (756 个 NPZ 文件)
├── val/    (252 个 NPZ 文件)
└── test/   (252 个 NPZ 文件)
```

每个 NPZ 文件包含：
| key | shape | dtype | 含义 |
|-----|-------|-------|------|
| `points` | (N, 3) | float32 | 点云坐标 (mm) |
| `labels` | (N,) | int64 | 标签 {0: 噪声, 1: 混凝土, 2: 钢筋, 3: 套筒} |
| `_geo.npy`（可选同名文件） | (N, 3) | float32 | 预计算几何特征（法线/曲率等，不存在时全零兜底） |

所有文件提前按构件类型分层随机拆分——每种构件类型内部独立划分，确保 train/val/test 的构件类型分布一致。

### 1.2 数据加载入口

`lib/pointnet2/dataset.py` → `create_dataloaders()`

```python
train_files, val_files, test_files = _list_npz_files(TRAIN_DIR)  # 硬编码路径
create_dataloaders(num_points=8192, batch_size=8, keep_ratio=0.4)
# → (make_train_loader, make_val_loader, make_test_loader)
```

调用时返回三个可调用对象，每次调用生成对应的 DataLoader。

---

## 2. 训练集采样策略

### 2.1 局部块采样（PatchBasedDataset）

训练集使用 **15cm 球查询**在原始点云中抽取局部块，解决全图一次性推断的内存问题，同时增加样本多样性。

每 epoch 重新生成的流程：

```
1. 文件列表随机打乱
2. 对每个文件:
   a. 从可用点池中随机选种子点
   b. 15cm ball query → 邻域候选点
   c. 如果候选点 ≥ 8192 → 随机抽 8192 个
   d. 如果候选点 < 8192 → KNN 补充到 8192 个
   e. 标记已用点不可再用
   f. 重复直到该文件点数耗尽
3. 统计所有块的钢筋比例 (rebar_ratio)
4. 筛选策略:
   - 所有含钢筋的块全部保留
   - 纯混凝土块只保留前 5 个
5. 打乱后作为本 epoch 的训练数据
```

筛选的目的是**缓解钢筋类别极度不平衡**（钢筋≈2%，混凝土≈95%）。

### 2.2 归一化

每个局部块提取后立即归一化：

```python
center = points.mean(axis=0)
points = points - center
scale = max(‖points‖₂)  # 单位: mm
points = points / (scale + 1e-8)  # 归一化到单位球
```

返回 4 值：

- `normed` — (8192, 3) 归一化坐标
- `geo` — (8192, 3) 几何特征（通常是全零）
- `labels` — (8192,) 每个点的标签
- `scale_mm` — 标量，原始物理尺度（用于动态半径）

### 2.3 双缓冲（DoubleBuffer）

训练集使用**生产者-消费者模式**避免训练等待数据生成：

```
┌─────────────┐     共享内存      ┌──────────────┐
│ Producer    │ ────────────────│ BufferDataset │
│ (子进程)    │  pts/geo/lbl    │ (训练循环)    │
│ 每 epoch    │  scale/n        │ 只读消费      │
│ 后台生成    │                 │ 零拷贝        │
└─────────────┘                 └──────────────┘
```

- 生产者循环 `regenerate → 写共享内存 → 等待消费 → regenerate`
- 消费者读取共享内存时不涉及序列化，零拷贝，不阻塞训练
- 训练循环对 DataLoader 的迭代触发下一轮生成的 ack

---

## 3. 验证/测试集采样策略

### 3.1 RandomValDataset

验证/测试集同样使用 15cm ball query + KNN 补充采样，但**不做钢筋筛选**，保持原始分布不变。

- 一次性预生成所有 patches（epoch 间固定不变）
- 确保不同模型在**完全相同的验证集**上对比公平

---

## 4. PointNet++ 模型架构

```
输入: (B, 8192, 3) + geo_features → (B, 8192, 6)

Encoder:
  SA1: FPS 512 点, ball query 80mm (物理), MLP[32,32,64]
       → (B, 512, 64)    in_ch=3(xyz)+3(geo)=6
  SA2: FPS 128 点, ball query 160mm (物理), MLP[64,64,128]
       → (B, 128, 128)   in_ch=64+3=67
  SA3: 全局, 无球查询, MLP[128,256,512]
       → (B, 1, 512)     in_ch=128+3=131

Decoder (Feature Propagation):
  FP3: 128 点 ← 1 点, 距离加权插值 + skip conn → MLP[256,256]
  FP2: 512 点 ← 128 点, 距离加权插值 + skip conn → MLP[128,128]
  FP1: 8192 点 ← 512 点, 距离加权插值 + skip conn → MLP[64,32]

Head: Conv1D(32,32) → BN → ReLU → Dropout(0.3) → Conv1D(32,4)
输出: (B, 8192, 4) 每个点 4 类 logits
```

### 关键特性

**动态物理半径**：传入 `scale_mm` 将物理半径映射到归一化空间。

```python
r1 = 80.0 / scale      # SA1: 物理 80mm → 归一化半径
r2 = 160.0 / scale     # SA2: 物理 160mm → 归一化半径
```

不同构件的物理尺度不同（柱 ~3000mm vs 梁 ~5000mm），统一物理半径确保不论构件大小，邻域范围在物理空间中一致。

---

## 5. 训练循环

### 5.1 超参数

| 参数         | 值   | 说明               |
| ------------ | ---- | ------------------ |
| NUM_POINTS   | 8192 | 每个样本点数       |
| BATCH_SIZE   | 8    | 批次大小           |
| NUM_EPOCHS   | 80   | 最大训练轮数       |
| LR           | 5e-4 | AdamW 初始学习率   |
| WEIGHT_DECAY | 1e-4 | 权重衰减           |
| keep_ratio   | 0.4  | 训练块筛选保留比例 |
| SEED         | 42   | 随机种子           |

### 5.2 损失函数：PerSampleWeightedCE

每样本自适应加权交叉熵，解决类别不平衡：

```python
w_c = N / (3 * n_c)                   # 混凝土/钢筋/套筒：按比例加权
w_c *= 0.5                             # 噪声降权（保留微弱梯度）
w_c = clamp(w_c, max=15.0)            # 防止极端加权
loss = Σ(w_c · CE_c) / N              # 除以 N 归一化
```

N = 8192，n_c = 当前样本中该类点数。

**为什么不用全局固定权重**：不同样本的钢筋比例差异很大（0%~15%），固定权重无法适应。

### 5.3 优化器与调度器

```python
optimizer = AdamW(lr=5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=80)
```

AdamW 比 Adam + L2 weight_decay 更稳定（解耦权重衰减与自适应学习率）。

### 5.4 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

防止梯度爆炸，尤其是早期训练阶段。

### 5.5 单 epoch 流程

```
for pts, feat, lbl, scale in train_loader:
    pts, feat, lbl, scale → GPU
    logits = model(pts, geo_features=feat, scale=scale)
    loss = criterion(logits.permute(0,2,1), lbl)
    if loss is NaN/Inf → skip (不更新)
    loss.backward()
    clip_grad_norm_(1.0)
    optimizer.step()
```

每个 epoch 结束后执行 `scheduler.step()`。

---

## 6. 验证循环

每个 epoch 验证一次：

```
model.eval()
for pts, feat, lbl, scale in val_loader:
    logits = model(pts, geo_features=feat, scale=scale)
    收集 logits 和 targets
all_preds = cat(all_preds)  # 全部验证集样本
all_targets = cat(all_targets)
```

### 6.1 评估指标

**IoU（交并比）**：

```python
IoU_c = |pred_c ∩ target_c| / |pred_c ∪ target_c|   # 每类
mIoU  = mean(IoU_噪声, IoU_混凝土, IoU_钢筋, IoU_套筒)
clean_mIoU = mean(IoU_混凝土, IoU_钢筋, IoU_套筒)    # 不含噪声
```

**Precision / Recall / F1**：

```python
precision_c = TP_c / (TP_c + FP_c)
recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * P_c * R_c / (P_c + R_c)
```

**Accuracy**：

```python
acc = (pred == target).float().mean()
```

### 6.2 最佳模型保存

验证 mIoU 超过历史最佳时保存 checkpoint：

```python
checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "val_miou": miou,
    "val_ious": val_ious,
    "val_prf": val_prf,
}
→ checkpoints/baseline_pointnet2/best_baseline_pointnet2.pth
```

---

## 7. 测试集评估

训练全部结束后加载最佳 checkpoint 在测试集上评估：

```
load best_baseline_pointnet2.pth
test_loss, test_ious, test_acc, test_prf, test_clean_miou = validate(test_loader)
```

测试集和验证集使用同样采样策略（不做筛选），确保结果可复现。

---

## 8. 输出文件

训练完成后 checkpoints 目录下生成：

```
checkpoints/
├── baseline_pointnet2/
│   ├── best_baseline_pointnet2.pth  ← 最佳模型权重
│   ├── history_baseline_pointnet2.json  ← 全 epoch 指标序列
│   ├── history_baseline_pointnet2.npz   ← 同上 (NPZ 格式)
│   └── summary_baseline_pointnet2.json  ← 实验汇总
```

### summary JSON 结构

```json
{
  "tag": "baseline_pointnet2",
  "desc": "PointNet++ + loss + 数据预处理",
  "best_epoch": 62,
  "best_val_miou": 0.853,
  "best_val_noise": 0.421,
  "best_val_concrete": 0.944,
  "best_val_rebar": 0.852,
  "best_val_sleeve": 0.863,
  "best_val_clean_miou": 0.886,
  "test_loss": 0.023,
  "test_acc": 0.958,
  "iou/噪声": 0.415,
  "iou/混凝土": 0.940,
  "iou/钢筋": 0.848,
  "iou/套筒": 0.859,
  "iou/mIoU": 0.766,
  "prf/噪声/precision": 0.832,
  "prf/噪声/recall": 0.711,
  ...
}
```

---

## 9. 完整流水线图

```
mock/inuse/{train,val,test}/*.npz
         │
         ▼
  create_dataloaders()
         │
         ├── train: 生产者进程 (子进程)
         │       │
         │       ├── 加载 NPZ → 缓存 KDTrees
         │       ├── 每 epoch:
         │       │     ├── 15cm ball query → patches
         │       │     ├── 按钢筋比例筛选 (keep_ratio=0.4)
         │       │     ├── 归一化到单位球
         │       │     └── 写入共享内存
         │       │
         │       └── 消费者: BufferDataset
         │             → DataLoader(batch_size=8, shuffle=True)
         │
         ├── val: 同上, keep_max_zero=-1 (不筛选)
         │     → DataLoader(batch_size=8, shuffle=False)
         │
         └── test: 同上
               → DataLoader(batch_size=8, shuffle=False)

  训练循环 (80 epochs)
         │
         ├── 前向: PointNet++
         │     SA1 → SA2 → SA3 → FP3 → FP2 → FP1 → Head
         │
         ├── 损失: PerSampleWeightedCE
         │
         ├── 反向: AdamW(lr=5e-4) + clip_grad_norm(1.0)
         │
         └── 调度: CosineAnnealingLR

  每 epoch 结束 → 验证 → 保存最佳 checkpoint

  80 epochs 结束 → 测试集评估 → 输出 summary JSON
```
