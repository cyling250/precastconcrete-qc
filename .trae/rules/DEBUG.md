# BUG 记录

## 级联管线相关

### BUG-024: Phase 1 (S1) 使用自定义 Stage1Model 导致 3 类性能劣于 A1 的 4 类

- **场景**: `experiment_phase1.py` 级联管线 S1 三分类训练
- **根因**: 原始 Phase 1 使用自定义 cascade 架构，而非标准 PointNet++。架构设计和初始化策略不同，导致 3 类分割效果反而不如 4 类
- **表现**: S1 噪声 IoU=0.77, 混凝土 IoU=0.97, 金属 IoU=0.88；而 A1 4 分类套筒 IoU=0.80, 钢筋 IoU=0.89
- **修复**: 重构为 `PointNetPlusPlus(num_classes=3, in_channels=3)` + `scale=None` + `nn.CrossEntropyLoss()` + 标签映射
- **教训**: 级联管线 S1 应复用已验证有效的基线架构，不要设计独立模型

### BUG-023: 级联管线 V3 Stage 1 head 随机初始化导致 Stage 2 永不触发

- **场景**: Phase 1 + Phase 2 级联管线 V3 推理
- **根因**: `best_s1.pth` 只保存了 backbone 权重，未保存 head。加载时 head 保持随机初始化，预测 metal 点数量为 0 → Stage 2 永不执行 → rebar/sleeve IoU=0
- **表现**: V3 在 80 epoch 中仅在第 5 epoch 记录到 mIoU=0.112，其余 epoch mIoU=0。V2（不使用 S1 head）正常 mIoU=0.838
- **修复**: 保存完整 `model.state_dict()`，加载时恢复所有模块权重
- **教训**: 多模块模型的 checkpoint 必须保存所有模块权重

### BUG-022: PointNeXt 套筒梯度饥饿 (class 3 logits 系统性被抑制 → IoU=0)

- **场景**: PointNeXt 训练，class 3 (套筒) 始终不被预测
- **根因**: 编码器-解码器结构导致 class 3 logits 在所有点上系统性低于其他类。梯度饥饿效应——套筒 logit 远低于其他类 → softmax 概率→0 → 交叉熵梯度→0 → logit 几乎不提升
- **修复**: 双重修复：(1) head 最后一层 `Conv1d(out=4)` 的 `bias[3] += 3.0`；(2) `PerSampleWeightedCE` 中添加 `sleeve_weight=3.0` 参数
- **教训**: 稀有类 + 深度架构可能产生梯度饥饿。对稀有类输出 bias 加正偏置可打破恶性循环

## 数据生成相关

### BUG-021: 矩形柱端面坐标错位（套筒/箍筋/钢筋均偏离正确位置）

- **场景**: `_builder.py` 的 `build_rectangular_column`
- **根因**: `sample_concrete_prism` 居中在 z=0（端面在 ±h/2），但 builder 中全部使用 (0, h) 而非 (-h/2, h/2)
- **修复**: 统一替换：`stirrups/rebars/sleeve/hole` 全部使用 ±h/2
- **教训**: 所有关联操作的坐标范围必须与混凝土棱柱的居中对齐一致

### BUG-020: 凹角处纵筋重叠（shapely buffer 圆弧伪影）

- **场景**: `_builder.py` 的 `_get_rebar_positions_polygon` 中 L 形柱/T 形梁截面凹角
- **根因**: `shapely.buffer(-offset, resolution=8)` 内缩多边形时，凹角处产生 8 个间距 5mm 的圆弧顶点，全部被当作独立纵筋位置
- **修复**: 增加相邻顶点合并步骤，跳过间距 `< cover*0.5` 的连续顶点
- **教训**: Shapely buffer 在凹角处会产生圆弧伪影，不能直接将其顶点作为特征点

### BUG-019: 箍筋坐标轴与构件轴向不匹配（梁箍筋沿 Z 而非 X）

- **场景**: `_rebar.py` 的箍筋函数被梁 builder 调用时
- **根因**: 箍筋函数假设截面在 XY 平面（柱坐标系），但梁截面在 YZ 平面，长度沿 X 轴
- **修复**: 给箍筋函数增加 `axis` 参数（0=X, 2=Z 默认），梁 builder 传入 `axis=0`
- **教训**: 几何库函数的轴向假设必须与调用方的坐标系对齐

### BUG-018: 同一构件同时包含纵筋和套筒（违反物理约束）

- **场景**: `_builder.py` 中 `with_sleeve=True` 的设计
- **根因**: 布尔参数导致同一构件同时出现纵筋 (label=2) 和套筒 (label=3)，实际二者不可并存
- **修复**: 替换为 `variant` 参数（`"rebar"` 或 `"sleeve"`），数据集生成各 50%
- **教训**: 二选一的配筋方式更符合物理世界语义

### BUG-017: 套筒轴向与构件轴向不匹配（梁套筒沿 Z 而非 X）

- **场景**: `_builder.py` 中梁构件的套筒生成
- **根因**: `generate_sleeve` 固定沿 Z 轴生成圆柱，但梁的长度轴是 X
- **修复**: 增加 `axis` 参数，梁用 axis=0，柱用 axis=2
- **教训**: 水平（梁）和竖直（柱）构件共用代码时，轴向参数化是必要设计

### BUG-016: `_sample_beam_cross_section` 传参错位

- **场景**: `_builder.py` 的端面采样
- **根因**: `sample_polygon_surface` 第二个参数是 `density`，但传入了 `n_endface`（点数），导致 `area * n_endface` 达到百万级
- **修复**: 用 `endface_density = n_endface / area` 替代直接传 `n_endface`
- **教训**: 函数接口的参数语义必须严格匹配

### BUG-013: L 形叠合梁混凝土分为两截（预制与叠合层间的间隙）

- **场景**: `build_l_composite_beam` 中叠合层 Y 偏移计算
- **根因**: 居中后截面顶部 Y 不再是 wb/2，但叠合层偏移沿用原始值，产生 ~23mm 间隙
- **修复**: 用居中后截面的实际最大 Y 替代硬编码的 wb/2
- **教训**: 居中后的几何坐标必须通过实际计算获取

### BUG-008: T 形梁纵向钢筋数量不足

- **场景**: `generate_t_beams` 中 `rebar_count` 参数
- **根因**: `rebar_count = rng.integers(2, 8)` 范围过低
- **修复**: 提升至 `rng.integers(4, 12)`
- **教训**: 合成数据参数应参考实际工程实践

### BUG-007: 矩形截面梁/柱缺少箍筋

- **场景**: `build_rectangular_column`
- **根因**: 初始设计只考虑了纵向钢筋，没有为矩形截面添加箍筋
- **修复**: 新增 `_generate_stirrups_rect` 函数生成矩形环箍筋
- **教训**: 所有混凝土构件的钢筋都应包含纵向筋 + 箍筋

### BUG-005: 箍筋应露出混凝土表面、形成闭环（装配式梁特征）

- **场景**: 装配式 T 形梁的箍筋生成
- **根因**: 箍筋完全埋入混凝土保护层内 + 只有底+两侧三段 (open-U)
- **修复**: 增加 `protrusion` 参数 (80~150mm)，箍筋段从 3 段改为 4 段闭环
- **教训**: 装配式构件的钢筋构造与现浇构件有本质区别

### BUG-002: T 形梁箍筋 Z 坐标计算错误（箍筋不生成）

- **场景**: `build_t_beam` 中箍筋 Z 坐标计算
- **根因**: `z_web_top = -hf / 2` 是错误经验值，与实际居中后多边形 Z 坐标不匹配
- **修复**: 替换为 `z_concrete_top = h - center[1]`
- **教训**: 居中坐标系的几何计算必须基于原始坐标减去中心偏移

### BUG-001: T 形梁纵向钢筋 Y 坐标偏移（钢筋跑出混凝土外）

- **场景**: 合成数据生成器 `build_t_beam`
- **根因**: 多边形居中后钢筋 Y 坐标未同步减去 `center[0]`
- **修复**: 钢筋 Y 坐标计算中减去 `center[0]`
- **教训**: 多边形居中后所有关联元素的坐标都要同步转换

## 模型实现相关

### BUG-015: PointMLP 解码器跳跃连接通道计算错误

- **场景**: `pointmlp_model.py` 的 decoder 构建
- **根因**: `reversed(widths[:-1] + [in_channels])` 计算出错误的 skip 通道数
- **修复**: 改为 `list(reversed(widths[:-1])) + [widths[0]]`
- **教训**: 跳跃连接通道数必须与各阶段实际输出特征通道严格对应

### BUG-014: PointMLP `GeometricAffine` 输入通道数错误

- **场景**: `pointmlp_model.py` 的 `GeometricAffine` 模块
- **根因**: `__init__` 声明 `in_channels=in_ch*2+3`，但 `forward` 实际拼接 `in_ch+3` 通道
- **修复**: `nn.Conv2d(in_ch * 2 + 3, ...)` → `nn.Conv2d(in_ch + 3, ...)`
- **教训**: 模型实现必须与 forward 实际拼接的维度一致

## 参数调整总结 (按规范)

根据 JG/T 398-2019、JGJ 1-2014、GB 50010、GB 1499.2 等规范调整以下参数：

| 参数                           | 修改前                   | 修改后                            | 依据                            |
| ------------------------------ | ------------------------ | --------------------------------- | ------------------------------- |
| `SLEEVE_BOTTOM_DENSITY_FACTOR` | 0.7                      | 5.0                               | 底部点云太稀疏不可见            |
| `SLEEVE_MIN_COVER` (新增)      | 无                       | 20mm                              | JGJ 1-2014 §6.5.3               |
| `STIRRUP_DIAM_RANGE` (新增)    | 硬编码 `(6,10)`          | `(8, 12)`                         | 预制构件箍筋宜 ≥8mm             |
| 套筒保护层计算                 | 固定 `CONCRETE_COVER=25` | 动态 `sid/2+6+20`                 | 套筒外径内缩量 >= 外半径+保护层 |
| 矩形柱端面坐标                 | `(0, h)`                 | `(-h/2, h/2)`                     | BUG-021 修复                    |
| `REBAR_DIAM_RANGE`             | 连续范围 (16,32)         | 离散标准值 [16,18,20,22,25,28,32] | GB 1499.2-2024                  |
| `SLEEVE_INNER_DIAM_RANGE`      | 连续范围 (25,55)         | `SLEEVE_SPEC_MAP` 映射表          | JGJ 355-2015 §3.1.2             |
| 钢筋-套筒匹配关系              | 无关联                   | 自动按钢筋直径查套筒规格          | 工程实际                        |
| `CONCRETE_ROUGHNESS` (新增)    | 完美光滑                 | σ=0.4mm 高斯粗糙度                | 模拟真实混凝土表面              |
| `STANDARD_REBAR_DIAMETERS`     | 无                       | [16,18,20,22,25,28,32]            | GB 1499.2-2024 标准系列         |
| `SLEEVE_SPEC_MAP`              | 连续范围                 | 外径/内径/长度映射表              | JG/T 398-2019 + 厂商规格        |
