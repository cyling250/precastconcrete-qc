# 项目交接文档

> 本文档由 AI 维护，用于任务交接。任务完成后写入「历史任务」，未完成任务见「当前任务」。
> 最后更新：2026-10-09 22:20 · Agent_20261009213000

## 项目快照

> 维护智能体：创建 Agent_20260928163100 · 最后修改 Agent_20261008153000

- **项目名称**：precastconcrete-qc
- **一句话描述**：预制混凝土构件点云材料级分割（两阶段级联 PC-CascadeNet）的 JBE 稿件配套代码；投稿件（含 Cover letter）已备齐于 `E:\OneDrive\预制构件分割\JBE\`，仓库已完成结果数据出库与实验代码卫生整理（T-011）。
- **技术栈**：Python 3.10（uv 管理，`.venv` 在项目根）+ torch 2.12.1+cu130 + numpy / open3d / scipy / rich / shapely / matplotlib；GPU 为 RTX 4060 Ti 16 GB。
- **架构概览**：
  - `lib/pointcloud_generator`：参数化点云数据集生成器（7 构件 × 10 构型 × 2 配筋 × 3 密度 × 3 噪声 = 1260 样本，落盘 `mock/inuse/`）；
  - `lib/cascade` + `lib/backbones`：两阶段级联分割（S1 三类噪声/混凝土/金属 → S2 精分割）与 PointMLP/PointNet/PointNet++/PointNeXt 骨干；
  - `lib/baselines`：几何基线（PCA 描述子 + RANSAC）及诊断工具（`_dbg_*`/`diag_features` 为 README 记载的诊断工具，非废弃代码）；`lib/measurement`：分割→下游测量（箍筋/纵筋间距、套筒偏位）；
  - `experiments/`：所有实验入口脚本（主基线 phase1/2、消融 A 系列、敏感性 S 系列、几何基线 B 系列、机制与闭环 E 系列）。
  - 数据流：生成器 → `mock/inuse/{train,val,test}` → 训练（权重入 `out/checkpoints`）→ 评测/敏感性/测量闭环（结果入 `out/E*`、`out/sensitivity`）。
- **目录含义**（2026-10-08 数据/代码卫生整理后）：
  - `src/lib/` — 全部库代码（backbones / baselines / cascade / measurement / pointcloud_generator / utils）
  - `src/experiments/` — 实验入口脚本，统一命名 `experiment_<编号>_<名称>.py`；`_` 前缀为辅助模块（`_s1_trainer.py`/`_s2_trainer.py`/`_a2_trainer.py`/`_e5_deep_predict.py`/`_config.py`）。E/B 系列 `--out` 相对路径统一经 `lib.utils.anchor_project_path` 锚定项目根
  - `scripts/` — 已入库的可复用脚本（队列执行器 queue_runner、绘图 make_figures、数据复核 summarize_out_jsons/key_numbers/e4e5_breakdown/e5_nested_p90、CSV 导出 make_plot_csvs；说明见 `scripts/README.md`）
  - `out/` — 全部结果数据（checkpoints / E1|E3|E4|E5|B1 / sensitivity / plot_csvs / train_logs / dataset），只留本地，整体 gitignore，不入库
  - `out/checkpoints/` — 模型权重、history、summary（按实验编号分目录）
  - `out/train_logs/` — 训练/运行日志（`00_…`~`21_…` 编号）、`progress.md` 进度史、`_runner_status.json` 队列终态
  - `out/E1|E3|E4|E5|B1/` — 各机制/对照实验结果 JSON（终版，改稿数字的依据）
  - `out/sensitivity/` — S1–S4 敏感性实验输出（S3/S4 新旧口径各一份，`_allkeys` 后缀 = 旧口径泄漏对照）
  - `out/plot_csvs/` — `scripts/make_plot_csvs.py` 导出的绘图 CSV（原 JBE修改包/绘图数据 的新家）
  - `out/dataset/` — `split_test_keys.json`（14 个测试划分键）
  - `out/figures/` — **已删除**（底稿图与作废的 Arial 风格图；权威图集在 `E:\OneDrive\预制构件分割\JBE\figure\`）
  - `docs/` — 仅本交接文档；稿件修改指导类文件已全部移出（2026-10-09，git 历史与 `F:\pcqc_backup_20260928\docs_deleted_20261008|09\` 可寻）
  - `mock/inuse/` — 生成数据集（6.9 GB，gitignore；`train/` 下 `.gpu_cache_*.pt` 为训练加速缓存，勿删）；`mock/origin`（干净样本）、`mock/add_noise`（加噪中间层）已删（可由生成器再造，运行时只读 inuse）
  - `temp_scripts/` — AI 临时脚本（gitignore；当前存 docx 提取与 Cover letter 更新 3 个小脚本）
  - `temp/` — 会话级中转目录（gitignore，用完即清）
  - `E:\OneDrive\预制构件分割\JBE\` — 投稿件目录（Manuscript + 5 个附件 + `figure\` 权威图集 18 个 TIFF + Cover letter new.docx）；**JBE修改包 已按用户要求删除**
  - `F:\pcqc_backup_20260928\` — **项目外**重组前全量备份（含被弃用的 `deprecated_s2_cache/`、迁移清单 `migration_report.json`、`docs_deleted_20261008/`）
- **常用命令**：
  - 安装依赖：`uv sync`（在项目根；torch 走 cu130 索引）
  - 跑单个实验：`cd src` 后 `<root>/.venv/Scripts/python.exe -m experiments.experiment_xxx`（模块搜索靠 cwd=src，路径常量已绝对化、与 cwd 无关）
  - 队列执行器（复用）：`"F:\precastconcrete-qc\.venv\Scripts\python.exe" "F:\precastconcrete-qc\scripts\queue_runner.py"`（幂等；当前 9/9 全部 DONE）
  - 队列状态：读 `out/train_logs/_runner_status.json`；进度史：`out/train_logs/progress.md`

## 历史任务

> 按任务组织（不是按对话），最新的在最上面。编号从 T-001 递增，永不复用。

> 维护智能体：创建 Agent_20260928163100 · 最后修改 Agent_20261009213000

### T-011 结果数据出库 + 实验代码卫生整理（投稿前定稿）
- **完成日期**：2026-10-09
- **执行智能体**：Agent_20261009213000
- **背景**：用户要求仓库只保留代码与结果数据：temp 与文献书写类文件清理，结果数据不入 git，experiments 的二次加工痕迹清理并规整实验命名；`E:\OneDrive\预制构件分割\JBE\` 不动。
- **文件清理**：删 `temp/`（jbe 手稿提取件、jbe_review）与 `temp_scripts/out/` 手稿副本（v2–v6 docx+md，约 130 MB）；删 docs 两份改稿指导（《稿件逐段修改指导》《支撑材料补充修改意见》，先备份 `F:\pcqc_backup_20260928\docs_deleted_20261009\`）；docs/ 只剩本交接文档。
- **git 出库**：`out/` 5 个跟踪文件（B1 JSON、split_test_keys、train_logs 三件）`git rm --cached`；`.gitignore` 以 `out/` 整目录取代 `out/checkpoints/`，新增 `temp/`，清掉重组前遗留的 `experiments/checkpoints` 等无效行。
- **experiments 痕迹清理**：5 个 E/B 脚本里复制粘贴的 4 行 `--out` 锚定补丁（P-006 遗留）收敛为 `lib.utils.anchor_project_path`（新增，`__init__` 已导出），各脚本只留一行调用；`experiment_s4_noise.py` 的 `--model m3` 分支重构为"模型先加载一次、统一循环"，产出文件名不变（`S4_summary*.json`）；`experiment_e5_deep.py` 实为 E5 的辅助模块而非实验入口，改名 `_e5_deep_predict.py` 归入 `_` 前缀辅助模块 family（e5_measurement 两处导入已同步）。phase1/phase2 命名保留：编号本身有含义（阶段1/阶段2 主基线），且与 checkpoint 目录 `phase*_baseline_*`、历史日志强耦合。
- **验证**：src 59 个 .py 全部字节编译通过；queue_runner 引用的 7 个模块名全部存在；8 个实验入口 `--help` 导入链全通；`anchor_project_path` 与原补丁公式逐字节等价。
- **相关任务**：[T-010](#t-010-第二轮核查用户应用修改后题注文字数字三线并行审查)、[P-006](#p-006-e-系列---out-相对-cwd-解析错位)、[P-007](#p-007-gitignore-忽略-scripts-差点重演-p-001)

### T-010 第二轮核查（用户应用修改后）：题注/文字/数字三线并行审查
- **完成日期**：2026-10-08
- **执行智能体**：Agent_20261008153000
- **背景**：用户按 T-009 清单改完一遍 Word，要求复核题注正确性、humanize-writing 合规性与表述错误；由 3 个子代理（视觉验收 / humanize 文字审查 / 技术数字核查）+ 本体复核完成。
- **已确认改好**：图注 C1–C5 全部到位（Fig 13 分述 (a)(b)+注明几何基线缺席+同页）；B1/B2/B3/B5/B6 应用；摘要 245 词。
- **新发现必修**：① 公式显示编号跳过 (7)（显示 1–6、8–11），正文 Eq.(7)–(10) 引用全部错一位；② **Figure 10/11 图注与图像错位**（噪声轴图无图注，kNN 图挂着 Figure 10 图注，Figure 11 图注悬空）；③ §4.2.3 PointNeXt "metal-class IoU 0.780 at pipeline level" 口径错（0.780 是 sleeve，rsIoU 应为 0.852）；④ Limitations 套筒 p90 区间 9.0–33.1 应为 5.8–33.1（low 档实测 5.8）；⑤ "monotonically" 不成立（S4 差值 t=0.1 回落，两处）；⑥ 噪声轴起点两处写 "clean scans"，代码证实 t=0=low 档（experiment_s4_noise.py `_interpolate_noise`）；⑦ Figure 7–17 图注+Table 6 表注全变蓝色。
- **上轮遗留未改**：B4（five→four 只删了 400，反而句内矛盾）、B7 [42]+Data availability、B8 AI 声明、A2 Heading4 `%4`、D1 fig07/08 y 轴标签。
- **humanize 结论**：AI 味中等偏高——4 处防御性辩解（M2 段最重、§4.5.2 收尾、§4.3、§4.6 开头）、冒号/分号后挂解释密集、§2.3 与 §2.3.1 近逐句重复、结论段指标倾泻；"statistically indistinguishable" 三处无检验支撑。
- **产出**：`docs/稿件逐段修改指导.md` 重写为纯操作清单（A Word 全局 / B 正文按序 / C 图注重绑 / D Origin 图内 / E 投稿件 / F 自查，全部带替换文字）；数字核查全部对表 out/ 终版 JSON（Table 5/6/8 逐格、E3/E4/E5/S3/S4 引用值均复现一致）。
- **第三轮核对（2026-10-09，用户应用修改后）**：公式编号已连续 1–10、Fig 10/11 图注已重绑、§4.2.3/§4.3/M2 三段重写已入稿、B 系列大部分完成；**新发现回归**——§4.6.1 rsIoU 引用域刷新后暴露错链（书签指向 cmIoU 式，现显示 Eq. (9)，须重链式 (10)）；A2 Heading4 `%4`、A3 题注蓝色（样式级 0000FF）、B20 five→four、B23 [42]、B24 AI 声明仍未改。剩余清单已按第三轮结果重写（仅待改项）；humanize-writing 技能吸收新规则 11（证据指针写进句子，不缩在括号里）。
- **相关任务**：[T-009](#t-009-换图手稿审阅--jbe修改包删除--项目代码数据文档卫生整理)

### T-009 换图手稿审阅 + JBE修改包删除 + 项目代码/数据/文档卫生整理
- **完成日期**：2026-10-08
- **执行智能体**：Agent_20261008153000
- **手稿审阅结论**：17 张内嵌图与权威图集 MD5 逐字节一致（image13 = fig13+fig13b 合并版，自带 (a)(b) 标签，T-008 遗留③解决）；R1/R2/R5 已应用、标题已套多级编号。**用 Word COM 转 PDF 实测渲染**，产出剩余清单写入 `docs/稿件逐段修改指导.md`（第三轮版）：
  - 必修新发现：① Heading4 编号模板为 `%4`，四层小节渲染成孤立 "1/2/3/4"（应 `%1.%2.%3.%4`）；② §4.6.2 正文"five radius values …400 mm"与 Fig 15 仅 4 点（100–300）不符（旧图遗留，建议改文按图）；③ §4.2.2 "(c)–(g)" 应为 "(c)–(f)"（新 Fig 8 共 6 面板）；④ Fig 7/8(b) y 轴 "Clean mIoU" 与图注 cmIoU 矛盾（Origin 侧）；
  - 未变的旧账：§4.6.1 "Eq. (9)"→(10)（域指错书签，F9 无效）、Fig 9–13 图注半成品（"图注："前缀/缺右括号/未分述 (a)(b)/跨页断开）、R6/R8 病句、R9 [42]、R12 AI 声明、Table 8 Δ 符号。
- **清理执行**：删 `JBE修改包\`（指导副本与 docs 版一致、figures_py 为权威图集复制件、CSV 可由 make_plot_csvs 再生）；temp_scripts 清空，7 个有效脚本沉淀 `scripts/`（README 记用途，输出路径 make_plot_csvs 改 `out/plot_csvs/`，summarize 的稿件路径改可选参数，均冒烟通过）；删 `mock/origin`(0.8G)+`mock/add_noise`(2.1G)（仅生成器流水线引用，训练/评测只读 inuse）；删 `out/figures/`(12M 底稿+作废风格)；清 `__pycache__`；`.gitignore` 去掉 `scripts/` 忽略（防 P-001 重演）；删 docs 三份已完成使命的指导文件（未入库的两份先备份 `F:\pcqc_backup_20260928\docs_deleted_20261008\`）；重写《稿件逐段修改指导》为剩余操作清单。
- **核对过的非删除项**：`_dbg_*.py`/`diag_features`/`selftest` 是 baselines README 记载的诊断工具（套筒结论证据脚本），保留；`experiment_e5_deep.py` 是 E5 现役模块，保留；仓库卫生三件事（matplotlib/死代码/_filter 注释）早已完成，无需动。
- **相关任务**：[T-008](#t-008-国庆后出图确认-jbefigure-为权威完整图集10-08-上午重绘案作废)

### T-008 国庆后出图：确认 `JBE\figure\` 为权威完整图集（10-08 上午重绘案作废）
- **完成日期**：2026-10-08
- **执行智能体**：Agent_20260930（10-08 会话）
- **结果更正**：用户反馈正确风格是**原稿带网格风格**。核实假期会话（10-02 00:52）已完成全部图件并归档 `E:\OneDrive\预制构件分割\JBE\figure\`（fig01–fig17+fig13b 共 18 个 TIFF；fig07/08 Origin 重绘、cmIoU 统一口径；fig16/17 泄漏修正数据），逐张核验数据正确。**本会话上午重绘的 Arial 红蓝黑风格 9 张与全稿风格不符，已作废**（备份 `F:\precastconcrete-qc\out\figures\panels_v2\`+composites；OneDrive `成品图\` 已删除）；figures_py 6 张即该权威集的 fig09–13b。
- **遗留待办**：① fig07/08 (b) 面板 y 轴标签 "Clean mIoU" 建议在 Origin 改 "cmIoU"；② fig13+fig13b 插 Word 时并排加 (a)(b) 标签；③ 假期会话的绘图脚本 `script_20261002_make_fig9_13.py` 未落盘（README 有记载，文件丢失，P-001 同款坑）；④ README 引用的 `add_expierment\稿件修改意见.md` 在 JBE 目录未找到。
- **指导文档已更新**：§2 改为"插图对照表"（图集已齐）；新增 R15（§4.2.2 两处 "Figure 8 (e)" 应为 "(d)"）。
- **相关任务**：[T-007](#t-007-第二轮核对用户已应用一版修改后的剩余项清单)

### T-007 第二轮核对：用户已应用一版修改后的剩余项清单
- **完成日期**：2026-09-30
- **执行智能体**：Agent_20260930
- **结果**：重新提取 11:53 版 Manuscript.docx 逐段核对第一轮 74 条指令的应用情况——摘要/贡献段/2.2.3/Table 4 表注/Table 5/Table 7/Table 8/消融三段/4.2.3/4.3/4.5 正文/§4.7 正文/结论/Limitations 均已正确应用；**重写 `docs/稿件逐段修改指导.md` 为第二轮版**（R1–R14 剩余项 + F1–F9 出图清单）。
- **剩余必修**：R1 §4.2.1 新旧两段重复（旧段 0.638/0.896 未删）；R2 Table 6 整表未替换（仍旧口径）；R3/R4 §4.6 图号 9/10 与新图冲突（→14/15）；R4 §4.7 图仍是旧 Fig 11/12（正文已引 16/17）。
- **新发现**：§4.6.1 "as defined in Eq. (9)" 应为 (10)（rsIoU 定义式，原稿遗留）；结论开头病句 "a class-balanced strategy local patch sampling strategy"；`_s1_trainer.py:69` 已验证 cm_fn=cm_iou(混凝土,金属,金属)，**Fig 7b/8b 的 history cm 列就是统一三类口径，可直接画，无需重算**（比第一轮指引简化）。
- **图任务**：5 张直接用底稿图（Fig 9–13）；Fig 16 = fig11a+11b 合并双面板；**Fig 17(a) 总体面板需新画**（S4_summary.json 三条曲线）；Fig 7b/8b 用 history cm 重绘。
- **相关任务**：[T-006](#t-006-jbe-稿件逐段修改指导--jbe-篇幅核查)

### T-006 JBE 稿件逐段修改指导 + JBE 篇幅核查
- **完成日期**：2026-09-30
- **执行智能体**：Agent_20260930（新会话）
- **结果**：产出 `docs/稿件逐段修改指导.md`——对 `E:\OneDrive\预制构件分割\JBE\Manuscript.docx` 按段落给出判定（不动/微调/重写/新增）与操作，全部数字经 `out/` 终版 JSON 重新核对（复核脚本 `temp_scripts/script_20260930_10*.py`）。同时核对 JBE 官方 Guide for Authors：**正文无字数/页数上限**，摘要 ≤250 词（现稿 217 ✓）、Highlights 3–5 条×≤85 字符、小节须编号（现稿无编号）、研究数据 Option C 强制。
- **关键新发现（两份底稿均未覆盖）**：统一口径重评后 **M2（去类均衡采样）与 S1 基线打平**（E1：mIoU 0.899 vs 0.899、rs 0.901 vs 0.899），旧稿"M2 使 rsIoU 降 0.030"不成立——指导给出诚实重写方案并要求软化引言贡献三/结论 (2)。
- **底稿勘误 7 处**（指导 §2）：R1"M1 降 0.118"→0.099；R4 val 最优 r=40→**50mm**；"never yields sleeve"→max IoU 0.033；R2 泄漏差端点（最大 +0.013 在 t=0.8 非最强档）；R5 cm 反转仅在 t=1.0 非 t≈0.8；E5 low 档 3.1→3.0mm；R3 PointNeXt 存活率句的分级表述。
- **相关产出**：`docs/稿件逐段修改指导.md`（主交付物，替代未落盘的《稿件修改文字_终稿》）。

### T-005 成稿底稿：8 张图 + 论文修改稿 MD
- **完成日期**：2026-09-29
- **执行智能体**：Agent_20260928163100
- **结果**：补跑 M3 单阶段 11 档噪声轴（S4 新增 `--model m3`，产出 `out/sensitivity/S4_summary_m3.json`）；绘制 8 张底稿图（`out/figures/`，scientific-plotting 规范，TIFF 600dpi）；写成稿 `docs/论文修改稿_补充实验.md`（R1–R6 六节英文正文 + 中文批注 + 重算表格）。
- **核心新发现**：级联对单阶段的 mIoU 优势随噪声单调扩大（+0.010 干净 → +0.027 t=1.0），cm 口径强噪声端反转（0.869 vs 0.863）；E5 分档拆解显示 medium 工况下级联纵筋误差 5.9mm（±10mm 容差内）vs 单阶段 15.3mm（超差）、套筒偏位 11.8 vs 72.1mm——级联叙事从"分割指标"迁移到"噪声轴 + 测量容差可靠性"。E3 修正认识：三骨干逐级 FPS 存活率几乎相同，PointNeXt 的差异在下采样链长度（128→32 归零金属类），显存解释不成立（9.3GB < 16GB）。
- **关键改动**：`src/experiments/experiment_s4_noise.py`（--model 参数、_load_m3/_evaluate_m3、None 安全打印、先落盘后打印）；`temp_scripts/script_20260929_173000_make_figures.py`（绘图脚本）。
- **相关坑**：[P-006](#p-006-e-系列---out-相对-cwd-解析错位)

### T-004 补充实验队列全部完成（9/9）
- **完成日期**：2026-09-29
- **执行智能体**：Agent_20260928163100
- **结果**：执行器 v2 于 09-29 14:02 以 exit 0 正常退出，9 步全部 OK/SKIPPED_DONE。关键产出与数字：
  - `a5_wce`（M5 消融）：测试 mIoU 0.864 / cm 0.887，M5 落定（`out/checkpoints/ablation_A5_wce/summary.json`）；
  - `E4` 几何基线 val 标定+test 报告（`out/E4/b2_*.json`，metal IoU 0.326）；`E3` FPS 存活率（`out/E3/`，pointnet++ 深层稀疏类存活率 ~0.0001，机制证据）；`E1` 统一 cmIoU 口径 Table 5/6/8 重算（`out/E1/E1_tables.json`）；
  - `S3/S4` 修正口径 + `--all-keys` 泄漏对照（`out/sensitivity/`）：修正口径整体略保守（S3 差 0.002–0.007，S4 差 0.002–0.013），泄漏修复后结论依然成立；
  - `E5` 分割→测量闭环（`out/E5/`）：级联纵筋间距误差 p90=1.86mm（84% 构件有效）vs 几何基线 p90=96mm（45%）；套筒匹配 14/16 vs 几何基线 0/16；几何基线仅在无遮挡的发夹筋上占优（误差 ~0 vs 级联 ~19.5mm）——是讨论素材。
- **关键改动**：E/B 系列 5 个脚本打 `--out` 锚定补丁（见 P-006）；队列运维记录见 `out/train_logs/progress.md`。
- **相关坑**：[P-006](#p-006-e-系列---out-相对-cwd-解析错位)

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

> 维护智能体：创建 Agent_20260928163100 · 最后修改 Agent_20261009213000

<!-- 当前任务完成后：整理为历史任务条目移入上方，然后恢复此占位 -->

**等待用户向 JBE 提交投稿件**（全部材料已备齐于 `E:\OneDrive\预制构件分割\JBE\`：Manuscript、Title Page、Highlights、Declaration of Interest、Supplementary material、Cover letter new.docx）。投稿后进入审稿意见处理阶段；仓库侧无未完成事项。

## 踩坑记录

> 每条坑必须链接到产生它的历史任务条目。

> 维护智能体：创建 Agent_20260928163100 · 最后修改 Agent_20261008153000

### P-007 .gitignore 忽略 scripts/ 差点重演 P-001
- **关联任务**：[T-009](#t-009-换图手稿审阅--jbe修改包删除--项目代码数据文档卫生整理)、[P-001](#p-001-队列脚本未入库丢失)
- **记录智能体**：Agent_20261008153000
- **现象**：整理脚本时发现 `.gitignore` 一直忽略整个 `scripts/` 目录——脚本即使按规范放进去也不会进版本库。
- **原因**：早期模板把 scripts/ 与 temp_scripts/ 一并忽略；P-001（队列脚本未入库丢失）的根因之一并未真正消除。
- **解决**：去掉 `scripts/` 忽略行（temp_scripts/ 保持忽略）；7 个有效脚本沉淀入 scripts/ 后应随下次提交入库。

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

### P-006 E 系列 --out 相对 cwd 解析错位
- **关联任务**：[T-004](#t-004-补充实验队列全部完成99)、[T-003](#t-003-项目目录重组src--out--docs-风格)
- **记录智能体**：Agent_20260928163100
- **现象**：E4 脚本 rc=0 正常跑完 56 分钟，但执行器判定 FAIL——结果写到了 `src/out/E4/` 而非 `out/E4/`。
- **原因**：脚本 `--out out/E4` 是相对路径，相对 cwd（重组后实验步 cwd=src）解析；重组时只改了默认值字符串，没有处理运行时相对路径。
- **解决**：5 个 E/B 脚本在 parse_args 后插入"相对路径锚定项目根"补丁（`temp_scripts/script_20260929_053000_fix_out_anchor.py`）；已错位的 E4/E3 结果手动搬运归位，执行器重试轮识别为 SKIPPED_DONE，未浪费重跑。教训：**凡是执行器以 cwd=X 运行、而产物路径写进命令行的场景，脚本内必须锚定绝对路径**。

## 修改台账

> 每次写入本档的区块级操作在此登记，区块归属以此表为准。
> 智能体编号 = `Agent_yyyyMMddHHmmss`，时间戳为该智能体在当前项目完成的第一个任务的完成时刻；编号规则实施前的历史内容标注"未登记"。

| 时间 | 智能体 | 板块 | 操作 | 摘要 |
|---|---|---|---|---|
| 2026-10-09 22:20 | Agent_20261009213000 | 全部 | 修订 | 按用户澄清，双语化改施于根目录 README（README.md 英文 / README.zh-CN.md 中文）；本档收回单语中文（开发用），保留 P-001 标题补回与 T-007→T-008 坏链修正 |
| 2026-10-09 22:00 | Agent_20261009213000 | 全部 | 追加/修改 | T-011：删 temp/ 与手稿副本、docs 两份改稿指导（先备份）；out/ 整体退出 git 并 gitignore；5 处 --out 锚定补丁收敛为 lib.utils.anchor_project_path；s4 m3 分支重构；experiment_e5_deep → _e5_deep_predict；scripts/ 首次入库 |
| 2026-10-08 15:40 | Agent_20261008153000 | 全部 | 追加/清理 | T-009：换图审阅（MD5+PDF 渲染），重写《稿件逐段修改指导》为剩余清单；删 JBE修改包、mock/origin+add_noise、out/figures、3 份旧指导文档；temp_scripts 沉淀 7 脚本入 scripts/ 并冒烟；.gitignore 去掉 scripts/ 忽略；新增 P-007 |
| 2026-09-30 15:50 | Agent_20260930 | 台账 | 追加 | 生成 `docs/JBE修改包/`（README + 逐段指导副本 + 9 个绘图 CSV，脚本 `temp_scripts/script_20260930_153000_make_csvs.py`），并整体复制到 `E:\OneDrive\预制构件分割\JBE\JBE修改包\`；phase2 history 字段为 val_cm/val_rs_iou（与 phase1 的 cm 不同名） |
| 2026-09-30 14:30 | Agent_20260930 | 历史任务/当前任务/台账 | 追加 | T-007：第二轮核对，指导文档重写为剩余项版（R1–R14 + 出图 F1–F9）；发现 §4.2.1 重复段、Table 6 未换、§4.6 图号冲突、Fig 17(a) 待新画；验证 Fig 7b/8b 可直接用 history cm |
| 2026-09-30 10:50 | Agent_20260930 | 历史任务/当前任务/台账 | 追加 | T-006：逐段修改指导 `docs/稿件逐段修改指导.md`（含 7 处底稿勘误、M2 打平新发现、JBE 篇幅核查：正文无上限/摘要 250 词）；当前任务改指向该指导 |
| 2026-09-29 19:40 | Agent_20260928163100 | 当前任务/台账 | 追加 | 按 humanize-writing 规范完成文字终稿 `docs/稿件修改文字_终稿.md`（M1–M12 可逐段替换），定位 JBE 投稿口径 |
| 2026-09-29 14:10 | Agent_20260928163100 | 历史任务/当前任务/踩坑/台账 | 追加/结算 | 结算 T-004（队列 ALL DONE 9/9），当前任务改为出图+正文修改清单，新增 P-006 |
| 2026-09-28 17:55 | Agent_20260928163100 | 全部 | 创建 | 初版交接文档：快照按重组后结构刷新；T-001~T-003、P-001~P-005 |
