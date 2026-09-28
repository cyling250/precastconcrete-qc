# 项目交接文档

> 本文档由 AI 维护，用于任务交接。任务完成后写入「历史任务」，未完成任务见「当前任务」。
> 最后更新：2026-09-28 17:55 · Agent_20260928163100

## 项目快照

> 维护智能体：创建 Agent_20260928163100 · 最后修改 Agent_20260928163100

- **项目名称**：precastconcrete-qc
- **一句话描述**：预制混凝土构件点云材料级分割（两阶段级联 PC-CascadeNet）的 JBE 稿件修改配套代码；当前阶段是跑「审稿自查报告」（`docs/实测数据与修改意见.md`）要求的补充实验。
- **技术栈**：Python 3.10（uv 管理，`.venv` 在项目根）+ torch 2.12.1+cu130 + numpy / open3d / scipy / rich / shapely / matplotlib；GPU 为 RTX 4060 Ti 16 GB。
- **架构概览**：
  - `lib/pointcloud_generator`：参数化点云数据集生成器（7 构件 × 10 构型 × 2 配筋 × 3 密度 × 3 噪声 = 1260 样本，落盘 `mock/inuse/`）；
  - `lib/cascade` + `lib/backbones`：两阶段级联分割（S1 三类噪声/混凝土/金属 → S2 精分割）与 PointMLP/PointNet/PointNet++/PointNeXt 骨干；
  - `lib/baselines`：几何基线（PCA 描述子 + RANSAC）及调试工具；`lib/measurement`：分割→下游测量（箍筋/纵筋间距、套筒偏位）；
  - `experiments/`：所有实验入口脚本（主基线 phase1/2、消融 A 系列、敏感性 S 系列、几何基线 B 系列、机制与闭环 E 系列）。
  - 数据流：生成器 → `mock/inuse/{train,val,test}` → 训练（权重入 `out/checkpoints`）→ 评测/敏感性/测量闭环（结果入 `out/E*`、`out/sensitivity`）。
- **目录含义**（2026-09-28 重组后）：
  - `src/lib/` — 全部库代码（backbones / baselines / cascade / measurement / pointcloud_generator / utils）
  - `src/experiments/` — 实验脚本，统一命名 `experiment_<编号>_<名称>.py` + 训练器辅助模块 `_s1_trainer.py`/`_s2_trainer.py`/`_a2_trainer.py`/`_config.py`
  - `out/checkpoints/` — 模型权重、history、summary（按实验编号分目录；已被 gitignore）
  - `out/train_logs/` — 训练/运行日志（`00_…`~`21_…` 编号）、`progress.md` 进度史、`_runner_status.json` 队列状态
  - `out/E1|E3|E4|E5|B1/` — 各机制/对照实验结果 JSON
  - `out/sensitivity/` — S1–S4 敏感性实验输出（S3/S4 新旧口径各一份，`_allkeys` 后缀 = 旧口径泄漏对照）
  - `out/dataset/` — `split_test_keys.json`（14 个测试划分键，S3/S4 修复泄漏的依据）
  - `docs/` — 本交接文档、《实测数据与修改意见》（审稿自查报告）、《绘图指导》
  - `mock/` — 生成数据集（9.7 GB，gitignore；`train/` 下 `.gpu_cache_*.pt` 为训练加速缓存，勿删）
  - `temp_scripts/` — AI 临时脚本（gitignore），含当前活跃的**队列执行器** `script_20260928_174500_queue_runner.py`
  - `F:\pcqc_backup_20260928\` — **项目外**重组前全量备份（含被弃用的 `deprecated_s2_cache/` 训练缓存 ~2.2 GB、旧 `.trae/.vscode`、迁移清单 `migration_report.json`）
- **常用命令**：
  - 安装依赖：`uv sync`（在项目根；torch 走 cu130 索引）
  - 跑单个实验：`cd src` 后 `<root>/.venv/Scripts/python.exe -m experiments.experiment_xxx`（模块搜索靠 cwd=src，路径常量已绝对化、与 cwd 无关）
  - 启动补充实验队列：`"F:\precastconcrete-qc\.venv\Scripts\python.exe" "F:\precastconcrete-qc\temp_scripts\script_20260928_174500_queue_runner.py"`（幂等，可随时重启续跑）
  - 队列状态：读 `out/train_logs/_runner_status.json`；进度史：`out/train_logs/progress.md`

## 历史任务

> 按任务组织（不是按对话），最新的在最上面。编号从 T-001 递增，永不复用。

> 维护智能体：创建 Agent_20260928163100 · 最后修改 Agent_20260928163100

### T-003 项目目录重组（src/ + out/ + docs/ 风格）
- **完成日期**：2026-09-28
- **执行智能体**：Agent_20260928163100
- **结果**：根目录从 7 个杂项目录收敛为 `src/ out/ docs/ mock/ temp_scripts/`；所有代码路径常量绝对化（与 cwd 无关）；重组前全量备份至 `F:\pcqc_backup_20260928\`；7 项自动化验证全部通过；队列以 v2 执行器在新结构下恢复。
- **关键改动**：
  - `lib/` → `src/lib/`，`experiments/` → `src/experiments/`（`__pycache__` 清理）；
  - `experiments/checkpoints/` → `out/checkpoints/`（12 个 `s2_*.pt` 训练缓存 ~2.2 GB 判定弃用 → 备份 `deprecated_s2_cache/`）；
  - `reports/` 拆分：结果 JSON → `out/E3|B1|dataset/`，日志 → `out/train_logs/`（含 `E4_b2_run.log` → `14_E4_b2_run_partial.log`），文档 → `docs/`；
  - 删除 `.trae/`、`.vscode/`（用户要求，原件在备份）；
  - 路径修改：`src/lib/utils/common.py`（`_PROJECT_ROOT` 四层 dirname、新增 `OUT_BASE`/`SENSITIVITY_BASE`、`CHECKPOINT_BASE` → `out/checkpoints`）、`src/lib/pointcloud_generator/_config.py`（相对路径 → 绝对路径）、S1–S4 敏感性输出 → `out/sensitivity/`、E/B 系列 `--out` 默认值 → `out/`、`_dbg_summarize.py` 读 `out/B1/`；
  - `.gitignore` 增补 `out/checkpoints/`、`temp_scripts/`；
  - 队列执行器重写为 v2（cwd=src、新标记文件路径）。
- **相关坑**：[P-002](#p-002-_configpy-项目根层级数算错)、[P-003](#p-003-生成器路径常量依赖-cwd)、[P-005](#p-005-迁移骨架预建目录与整目录-move-冲突)

### T-002 补充实验队列恢复（第一次，重组前）
- **完成日期**：2026-09-28
- **执行智能体**：Agent_20260928163100
- **结果**：确认 13 步训练队列只剩 `a5_m5_wce`（9-23 中断于 ep50，断点完好）；重建队列执行器 v1 并后台启动 a5 续训；建立每 2 小时健康检查自动化。后因目录重组（T-003）被主动停止，由 T-003 的 v2 执行器接替。
- **关键改动**：`temp_scripts/script_20260928_162030_queue_runner.py`（v1，已被 v2 替代并移入备份）
- **相关坑**：[P-001](#p-001-队列脚本未入库丢失)、[P-004](#p-004-zcode-会话闪退致后台任务脱管)

### T-001 补充实验体系搭建与 13 步主训练队列（重组前历史，会话多段接力）
- **完成日期**：2026-09-23（训练队列中断时点）
- **执行智能体**：未登记（编号规则实施前的历史会话）
- **结果**：按审稿自查报告 2.1–2.5 节完成全部代码修复与 13 步训练队列的 12 步：phase1/phase2 基线 × {pointmlp, pointnet++, pointnext(b8 补跑), pointnet}、消融 A1a/A1b/A2(M3)/A4(focal) 全部产出权重与 summary；仅剩 step13 `a5_m5_wce`（中断于 ep50/120，最佳 val mIoU 0.846）。全程记录在 `out/train_logs/progress.md`（83 条巡检记录）。
- **关键改动**：S3/S4 增加测试键过滤（修评测集泄漏）与 `--all-keys` 泄漏对照参数；`_s1_trainer` 支持批大小参数（pointnext b8）；E1/E3/E5/B2/E5-deep 实验脚本与 `lib/measurement` 均为本阶段新增。
- **相关坑**：[P-001](#p-001-队列脚本未入库丢失)、[P-004](#p-004-zcode-会话闪退致后台任务脱管)

## 当前任务

> 维护智能体：创建 Agent_20260928163100 · 最后修改 Agent_20260928163100

<!-- 当前任务完成后：整理为历史任务条目移入上方，然后恢复此占位 -->

**补充实验队列自动推进中**（执行器 v2，pid 见 `out/train_logs/_runner_status.json`）：

| # | 步骤 | 状态 | 完成标记 |
|---|---|---|---|
| 13 | train_a5_wce（断点 ep39/120 续训，~9–11h） | RUNNING | `out/checkpoints/ablation_A5_wce/summary.json` |
| 14 | E4_b2_geo_valtest（CPU 数小时） | 排队 | `out/E4/b2_test_eval.json` |
| 15 | E3_fps_survival | 排队 | `out/E3/e3_fps_survival.json` |
| 16 | E1_tables（依赖 A5 权重） | 排队 | `out/E1/E1_tables.json` |
| 17/18 | S3_density 新旧口径 | 排队 | `out/sensitivity/S3_summary*.json` |
| 19/20 | S4_noise 新旧口径 | 排队 | `out/sensitivity/S4_summary*.json` |
| 21 | E5_measurement_deep | 排队 | `out/E5/e5_summary.json` |

- 每 2 小时健康检查自动化 `automation-0ef57d06` 会巡检、续跑；全链预计 1.5–2 天。
- 队列 ALL DONE 后的下一件事：按 `docs/绘图指导.md` 出图（Fig 11a/b、12a/b 需 S3/S4 修正口径数据 + E 系列结果），并按 `docs/实测数据与修改意见.md` 第 3 节动作清单改正文。
- 注意：`temp_scripts/` 与 `out/checkpoints/` 被 gitignore，**执行器脚本无版本保护**；如担心丢失可将其复制到版本库（备份目录里有副本）。

## 踩坑记录

> 每条坑必须链接到产生它的历史任务条目。

> 维护智能体：创建 Agent_20260928163100 · 最后修改 Agent_20260928163100

### P-001 队列脚本未入库丢失
- **关联任务**：[T-002](#t-002-补充实验队列恢复第一次重组前)、[T-001](#t-001-补充实验体系搭建与-13-步主训练队列重组前历史会话多段接力)
- **记录智能体**：Agent_20260928163100
- **现象**：恢复队列时发现 `train_queue.py`、`post_queue_runner.py`、`handover.md` 全部不在磁盘也不在 git 历史。
- **原因**：这些运维文件放在项目根且从未 `git add`；目录清理提交（c692f19 一类）波及后无法从版本库找回。
- **解决**：按 progress.md 记录重建执行器（v1→v2）；今后关键运维脚本要么纳入版本库、要么确保在项目外备份目录有副本。

### P-002 _config.py 项目根层级数算错
- **关联任务**：[T-003](#t-003-项目目录重组src--out--docs-风格)
- **记录智能体**：Agent_20260928163100
- **现象**：重组后 `_config.py` 的 `_PROJECT_ROOT` 解析为 `…\src`（应为项目根），`mock/inuse/test` 找不到。
- **原因**：从 `src/lib/pointcloud_generator/_config.py` 回项目根需要**四层** dirname，写成了三层；层级数按"源代码在根目录下第几层"的旧记忆来算就会错。
- **解决**：改为四层 dirname 并验证；验证脚本（`temp_scripts/script_20260928_173000_verify.py`）成功拦截，未流入运行阶段。

### P-003 生成器路径常量依赖 cwd
- **关联任务**：[T-003](#t-003-项目目录重组src--out--docs-风格)
- **记录智能体**：Agent_20260928163100
- **现象**：即使 common.py 已绝对化，`test_keys()` 仍找不到数据集。
- **原因**：`lib/pointcloud_generator/_config.py` 另有一套相对路径常量（`"mock/inuse/test"` 等），原项目靠 cwd=项目根掩盖了问题；运行 cwd 改为 src 后暴露。
- **解决**：该文件路径常量全部改为基于 `_PROJECT_ROOT` 的绝对路径；全库 grep 确认无其他 cwd 依赖（docstring 示例除外）。

### P-004 ZCode 会话闪退致后台任务脱管
- **关联任务**：[T-002](#t-002-补充实验队列恢复第一次重组前)、[T-001](#t-001-补充实验体系搭建与-13-步主训练队列重组前历史会话多段接力)
- **记录智能体**：Agent_20260928163100
- **现象**：历史会话中多次 ZCode 闪退/重载后，后台终端任务条目消失，训练进程或被连带杀死、或脱管存活但不可见。
- **原因**：后台任务绑定会话生命周期。
- **解决**：每 2 小时健康检查自动化兜底（存活→记录进度；死亡→断点重启）；训练本身有 epoch 级断点，损失有限。

### P-005 迁移骨架预建目录与整目录 move 冲突
- **关联任务**：[T-003](#t-003-项目目录重组src--out--docs-风格)
- **记录智能体**：Agent_20260928163100
- **现象**：迁移脚本首次运行在 `mv("experiments/checkpoints", "out/checkpoints")` 处失败。
- **原因**：骨架步骤预建了空 `out/checkpoints`，shutil.move 对"目标目录已存在"会把源目录塞进目标内部，保护逻辑正确拦截但脚本中断。
- **解决**：骨架不预建该目录、由 move 创建；重跑前确认遗留空目录已清理。

## 修改台账

> 每次写入本档的区块级操作在此登记，区块归属以此表为准。
> 智能体编号 = `Agent_yyyyMMddHHmmss`，时间戳为该智能体在当前项目完成的第一个任务的完成时刻；编号规则实施前的历史内容标注"未登记"。

| 时间 | 智能体 | 板块 | 操作 | 摘要 |
|---|---|---|---|---|
| 2026-09-28 17:55 | Agent_20260928163100 | 全部 | 创建 | 初版交接文档：快照按重组后结构刷新；T-001~T-003、P-001~P-005 |
