# PC-CascadeNet：预制混凝土构件点云材料级分割

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%20CUDA%2013.0-red)](https://pytorch.org/)

论文《Point-cloud-based material level segmentation of precast concrete components for automated quality inspection using a two-stage cascade network》（投稿于 *Journal of Building Engineering*）配套代码仓库。

**Languages / 语言：[English](README.md) | [中文](README.zh-CN.md)**

## 简介

预制混凝土（PC）构件的尺寸质量检验是装配式建造质量控制的前置环节，激光扫描已成为主要检测手段。要把扫描点云变成检验结论，需要先把点云分割成四个材料级类别——**混凝土、钢筋、灌浆套筒、噪声**——而这一步受制于三个难题：现场扫描噪声严重、类别极度不均衡（钢筋与套筒点稀少但结构上最关键）、带标注的真实数据稀缺。

本仓库提供三部分内容：

1. **PC-CascadeNet 两阶段级联分割网络**。第一阶段做粗糙三类分割（噪声 / 混凝土 / 钢筋-套筒并集）；对混凝土点自适应体素下采样后，第二阶段做精细四类分割。配合类均衡的局部补片采样（150 mm 球查询，覆盖 GB 50010-2010 规定的典型钢筋间距范围），保证稀少类别获得足够的训练信号。核心思路：占主导但几何上简单的混凝土类会干扰对稀少钢筋与套筒的学习，级联让每个阶段专注解决自己的子问题。
2. **参数化合成数据生成器**，以 JGJ 355-2015（钢筋灌浆套筒连接）与 GB 50010-2010（混凝土结构设计规范）为依据。它合成 **1,260 个带标注点云**：7 种 PC 构件 × 2 种配筋变体 × 3 档密度 × 3 档耦合噪声（六种耦合噪声类型模拟现场扫描），并保持各材料间点密度一致。
3. **完整实验体系**：骨干对比（PointMLP、PointNet、PointNet++、PointNeXt，级联两阶段各自对比）、消融实验（随机采样 / 去正样本均衡 / 单阶段 / focal loss / 加权交叉熵）、经验证集标定的经典几何基线（PCA 描述子 + RANSAC）、敏感性与鲁棒性分析（补片半径、体素尺寸、密度、噪声），以及**分割 → 测量的闭环实验**——按规范容差检验钢筋间距与套筒偏位。

## 主要结果

- 合成测试集上 **平均 IoU 91.8%**，在级联两个阶段均优于三种主流点云分割架构。
- 下游测量中，PC-CascadeNet 是**唯一在中等噪声以内将纵筋间距误差保持在 ±10 mm 安装容差内的受测变体**——分割增益能传导到实际检验。
- 测量闭环中对比经标定的几何基线：套筒匹配 14/16 构件 vs 0/16；纵筋间距误差 p90 = 1.86 mm（84% 构件有效）vs 96 mm（45% 有效）。

## 仓库结构

```
precastconcrete-qc/
├── src/
│   ├── lib/
│   │   ├── pointcloud_generator/   # 参数化数据集生成器（7 种构件）
│   │   ├── cascade/                # 两阶段级联模型（S1/S2）
│   │   ├── backbones/              # PointMLP / PointNet / PointNet++ / PointNeXt
│   │   ├── baselines/              # 几何基线（PCA + RANSAC）及诊断工具
│   │   ├── measurement/            # 分割 → 尺寸测量闭环
│   │   └── utils/                  # 公共常量、数据集、训练、指标
│   └── experiments/                # 实验入口脚本（见下表）
├── scripts/                        # 队列执行器、绘图、结果复核
├── docs/                           # 开发交接文档（中文）
├── mock/                           # 生成的数据集（不入库，本地约 6.9 GB）
└── out/                            # 权重、结果、日志（不入库，仅本地）
```

## 实验命名

| 前缀 | 实验族 | 脚本 |
|---|---|---|
| `phase1` / `phase2` | 主骨干基线（阶段 1 / 阶段 2） | `experiment_phase1_baseline.py`、`experiment_phase2_baseline.py` |
| `a1a`–`a5` | 消融实验（论文模型 M1–M5） | 随机采样、去正样本均衡、单阶段（M3）、focal loss、加权交叉熵 |
| `b1`、`b2` | 经典几何基线 | 全量扫掠（`out/B1`）、验证集标定 + 测试报告（`out/E4`） |
| `e1`、`e3`、`e5` | 机制与闭环 | 统一口径对比表、FPS 逐级存活率、分割→测量闭环 |
| `s1`–`s4` | 敏感性 / 鲁棒性 | 补片半径、体素尺寸、密度、噪声（11 档噪声轴；`--all-keys` = 旧口径泄漏对照） |

`src/experiments/` 下以 `_` 开头的文件是辅助模块（训练器、共享配置、全云逐点预测），不是实验入口。

## 快速开始

环境要求：Python 3.10，NVIDIA GPU（参考运行环境为 RTX 4060 Ti 16 GB）。

```bash
# 1. 安装依赖（项目根目录；torch 走 cu130 索引）
uv sync

# 2. 生成完整的 1,260 样本数据集到 mock/inuse/
cd src
../.venv/Scripts/python.exe -m lib.pointcloud_generator --batch

# 3. 训练两阶段主基线（换 --arch pointnet / pointnet++ / pointnext 重复）
../.venv/Scripts/python.exe -m experiments.experiment_phase1_baseline --arch pointmlp
../.venv/Scripts/python.exe -m experiments.experiment_phase2_baseline --arch pointmlp

# 4. 评测 / 复现论文数字（在 src/ 下运行，结果写入 out/）
../.venv/Scripts/python.exe -m experiments.experiment_e1_tables --out out/E1        # 对比表
../.venv/Scripts/python.exe -m experiments.experiment_s4_noise                      # 噪声鲁棒性
../.venv/Scripts/python.exe -m experiments.experiment_e5_measurement --out out/E5 --deep   # 测量闭环
```

消融、几何基线与其余敏感性脚本用法相同（`python -m experiments.experiment_<编号>_<名称>`）。`scripts/queue_runner.py` 可幂等地跑完整补充实验队列并断点续训。实验产物写入 `out/`、不入库；`mock/` 数据集由带种子的生成器复现。

## 数据可用性

支撑论文的数据集与源代码均在本仓库公开。由于带标注数据集由确定性参数化生成器产出，可用上述命令在本地复现，因此不以文件形式入库。

## 引用

如果本仓库对您的研究有帮助，请引用对应论文（录用后补充引用信息）。
