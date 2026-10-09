# scripts/ — 可复用脚本

从 `temp_scripts/` 沉淀的项目运维与数据工具脚本（2026-10-08 整理）。
统一用项目根 `.venv` 解释器运行；实验类模块跑法见 `docs/handover.md` 快照。

| 脚本 | 用途 | 典型调用 |
|---|---|---|
| `queue_runner.py` | 补充实验队列执行器（v2，幂等可续跑；9/9 已全部完成，留作以后加实验复用） | `.venv/Scripts/python.exe scripts/queue_runner.py` |
| `make_figures.py` | 底稿数据图绘制（scientific-plotting 规范，80×60mm、TIFF 600dpi，输出 `out/figures/`） | `.venv/Scripts/python.exe scripts/make_figures.py` |
| `summarize_out_jsons.py` | 汇总 `out/` 各补充实验 JSON 关键数字 + 统计摘要词数（JBE ≤250 词） | `.venv/Scripts/python.exe scripts/summarize_out_jsons.py` |
| `key_numbers.py` | 提取改稿对照的精确数字（S4/S3/M3 曲线端点、E3 存活率、E4/E5 关键指标） | `.venv/Scripts/python.exe scripts/key_numbers.py` |
| `e4e5_breakdown.py` | 重算 E4 几何基线 test 均值、E5 按噪声分档指标、E3 逐级存活率 | `.venv/Scripts/python.exe scripts/e4e5_breakdown.py` |
| `e5_nested_p90.py` | 解析 `out/E5/e5_measurement.json` 嵌套记录，按方法×噪声档重算 p90 与有效率 | `.venv/Scripts/python.exe scripts/e5_nested_p90.py` |
| `make_plot_csvs.py` | 从 `out/` 终版 JSON 导出全部绘图 CSV（utf-8-sig），输出 `out/plot_csvs/` | `.venv/Scripts/python.exe scripts/make_plot_csvs.py` |

注意事项：

- 这些脚本为项目专用（路径锚定 `F:\precastconcrete-qc`），不追求通用化。
- 一次性迁移/修复脚本（目录重组 backup/migrate/fix_outpaths/verify、--out 锚定补丁）
  与已作废的 Arial 风格重绘脚本已于 2026-10-08 清理；重组前旧件在 `F:\pcqc_backup_20260928\`。
- 权威图集在 `E:\OneDrive\预制构件分割\JBE\figure\`（fig01–fig17 + fig13b）；
  `out/figures/` 底稿图与作废风格图已随同清理删除。
