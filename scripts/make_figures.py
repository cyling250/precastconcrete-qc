# -*- coding: utf-8 -*-
"""
创建时间：2026-09-29 17:30:00
用途：为成稿 MD 绘制全部数据图（底稿版，用户后续会按 scientific-plotting 规范重绘）。
      遵循 scientific-plotting 技能：80×60mm、Arial 9pt、TIFF 600dpi、单轴、
      三色板（红 #C4323F / 蓝 #033250 / 黑 #000000），>3 序列用同色板+线型区分。
      输出到 out/figures/。
依赖：matplotlib（.venv 内）、sci_style 样式模块
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\35160\.zcode\skills\scientific-plotting\scripts")
import sci_style as ss
import numpy as np

ROOT = r"F:\precastconcrete-qc"
SENS = os.path.join(ROOT, "out", "sensitivity")
FIGS = os.path.join(ROOT, "out", "figures")
os.makedirs(FIGS, exist_ok=True)

ss.apply_style()

# ── 数据装载 ──
s3 = json.load(open(os.path.join(SENS, "S3_summary.json"), encoding="utf-8"))
s4 = json.load(open(os.path.join(SENS, "S4_summary.json"), encoding="utf-8"))
s4_m3_path = os.path.join(SENS, "S4_summary_m3.json")
s4_m3 = json.load(open(s4_m3_path, encoding="utf-8")) if os.path.exists(s4_m3_path) else None
e3 = json.load(open(os.path.join(ROOT, "out", "E3", "e3_fps_survival.json"), encoding="utf-8"))

# 几何基线 11 档 (B1 实测, 密度=medium, 4 类构件均值) —— docs/实测数据与修改意见.md 4.2
GEO_T = np.arange(0.0, 1.01, 0.1)
GEO_METAL = np.array([0.707, 0.676, 0.604, 0.579, 0.531, 0.473, 0.413, 0.335, 0.266, 0.227, 0.159])
# 邻域方式对照 (B1 实测 2.4c) —— kNN(k=24) vs 固定物理半径(40mm), 金属 IoU @ 3 密度
NB_DENS = np.array([0.006, 0.012, 0.020])
NB_KNN = np.array([0.597, 0.306, 0.116])
NB_RADIUS = np.array([0.690, 0.765, 0.557])
# E5 纵筋间距误差 p90 (out/E5/e5_measurement.json 分档统计)
E5_NOISE = ["Clean", "Low", "Medium", "High"]
E5_CASCADE = [2.21, 3.05, 5.87, 36.69]
E5_M3 = [3.71, 0.78, 15.27, 35.72]
E5_GEO = [27.90, 26.76, 195.70, 22.76]   # medium 超出 y 上限, 图内注记


def save(fig, name):
    ss.save_tiff(fig, os.path.join(FIGS, name))
    print(f"  [ok] {name}")


# ── Fig 11a: 密度鲁棒性 (总指标) ──
d = sorted(s3.values(), key=lambda r: r["density"])
x = np.array([r["density"] for r in d])
fig, ax = ss.new_figure()
ax.plot(x, [r["test_mIoU"] for r in d], color=ss.RED, lw=ss.LINE_W, label="mIoU (4-class)")
ax.plot(x, [r["test_cm"] for r in d], color=ss.BLUE, lw=ss.LINE_W, label="cmIoU (3-class)")
ax.plot(x, [r["test_rs_iou"] for r in d], color=ss.BLACK, lw=ss.LINE_W, label="rsIoU")
ax.set_xlabel("Sampling density (pts/mm$^2$)")
ax.set_ylabel("IoU (-)")
ax.legend(loc="lower right")
ss.cap_axes(ax, xlim=(0.006, 0.020), ylim=(0.80, 0.96))
save(fig, "fig11a_density_overall.tiff")

# ── Fig 11b: 密度鲁棒性 (逐类, 3 色+线型) ──
fig, ax = ss.new_figure()
ax.plot(x, [r["iou/混凝土"] for r in d], color=ss.RED, lw=ss.LINE_W, label="Concrete")
ax.plot(x, [r["iou/钢筋"] for r in d], color=ss.BLUE, lw=ss.LINE_W, label="Rebar")
ax.plot(x, [r["iou/噪声"] for r in d], color=ss.BLACK, lw=ss.LINE_W, label="Noise")
ax.plot(x, [r["iou/套筒"] for r in d], color=ss.BLACK, lw=ss.LINE_W, ls=":", label="Sleeve")
ax.set_xlabel("Sampling density (pts/mm$^2$)")
ax.set_ylabel("IoU (-)")
ax.legend(loc="center right")
ss.cap_axes(ax, xlim=(0.006, 0.020), ylim=(0.70, 1.0))
save(fig, "fig11b_density_perclass.tiff")

# ── Fig 12a: 噪声退化对比 (级联 vs 几何基线, 金属类口径) ──
t = np.array([r["noise_intensity"] for r in sorted(s4.values(), key=lambda r: r["noise_intensity"])])
fig, ax = ss.new_figure()
ax.plot(t, [r["test_rs_iou"] for r in sorted(s4.values(), key=lambda r: r["noise_intensity"])],
        color=ss.RED, lw=ss.LINE_W, label="PC-CascadeNet")
ax.plot(GEO_T, GEO_METAL, color=ss.BLUE, lw=ss.LINE_W, label="Geometric baseline")
ax.set_xlabel("Noise intensity (-)")
ax.set_ylabel("Metal IoU (-)")
ax.legend(loc="lower left")
ss.cap_axes(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
save(fig, "fig12a_noise_cascade_vs_geo.tiff")

# ── Fig 12b: 级联噪声退化逐类 (3 色+线型) ──
fig, ax = ss.new_figure()
ax.plot(t, [r["iou/混凝土"] for r in sorted(s4.values(), key=lambda r: r["noise_intensity"])],
        color=ss.RED, lw=ss.LINE_W, label="Concrete")
ax.plot(t, [r["iou/钢筋"] for r in sorted(s4.values(), key=lambda r: r["noise_intensity"])],
        color=ss.BLUE, lw=ss.LINE_W, label="Rebar")
ax.plot(t, [r["iou/噪声"] for r in sorted(s4.values(), key=lambda r: r["noise_intensity"])],
        color=ss.BLACK, lw=ss.LINE_W, label="Noise")
ax.plot(t, [r["iou/套筒"] for r in sorted(s4.values(), key=lambda r: r["noise_intensity"])],
        color=ss.BLACK, lw=ss.LINE_W, ls=":", label="Sleeve")
ax.set_xlabel("Noise intensity (-)")
ax.set_ylabel("IoU (-)")
ax.legend(loc="lower left")
ss.cap_axes(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
save(fig, "fig12b_noise_perclass.tiff")

# ── Fig 12c: 级联 vs 单阶段 M3 (4 类 mIoU) —— 依赖 M3 数据 ──
if s4_m3 is not None:
    tm3 = np.array([r["noise_intensity"] for r in sorted(s4_m3.values(), key=lambda r: r["noise_intensity"])])
    fig, ax = ss.new_figure()
    ax.plot(t, [r["test_mIoU"] for r in sorted(s4.values(), key=lambda r: r["noise_intensity"])],
            color=ss.RED, lw=ss.LINE_W, label="PC-CascadeNet")
    ax.plot(tm3, [r["test_mIoU"] for r in sorted(s4_m3.values(), key=lambda r: r["noise_intensity"])],
            color=ss.BLACK, lw=ss.LINE_W, label="Single-stage (M3)")
    ax.set_xlabel("Noise intensity (-)")
    ax.set_ylabel("mIoU (-)")
    ax.legend(loc="lower left")
    ss.cap_axes(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    save(fig, "fig12c_noise_cascade_vs_m3.tiff")
else:
    print("  [skip] fig12c (M3 数据未就绪)")

# ── Fig E3: FPS 逐级钢筋点存活率 (3 骨干) ──
STAGE_LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
ARCH_COLOR = {"pointmlp": ss.RED, "pointnet++": ss.BLUE, "pointnext": ss.BLACK}
ARCH_LABEL = {"pointmlp": "PointMLP", "pointnet++": "PointNet++", "pointnext": "PointNeXt"}
STAGE_ORDER = {
    "pointmlp": ["8192->2048", "2048->512", "512->128"],
    "pointnet++": ["8192->512", "512->128", "128->1"],
    "pointnext": ["8192->2048", "2048->512", "512->128", "128->32"],
}
fig, ax = ss.new_figure()
for arch, color in ARCH_COLOR.items():
    seeds = e3["result"][arch]
    stages = STAGE_ORDER[arch]
    ys = [float(np.mean([s[st]["rebar"] for s in seeds])) for st in stages]
    xs = list(range(1, len(stages) + 1))
    ax.plot(xs, ys, color=color, lw=ss.LINE_W, marker="o", markersize=3,
            label=ARCH_LABEL[arch])
ax.set_xticks(range(1, 5))
ax.set_xticklabels(STAGE_LABELS)
ax.set_xlabel("Successive FPS stages")
ax.set_ylabel("Rebar survival rate (-)")
ax.legend(loc="upper right")
ss.cap_axes(ax, xlim=(0.6, 4.4), ylim=(0.0, 0.7), cap_x=False)
save(fig, "fig_e3_fps_rebar_survival.tiff")

# ── Fig E5: 纵筋间距测量误差 p90 按噪声档 (级联/单阶段/几何) ──
fig, ax = ss.new_figure()
xpos = np.arange(1, 5)
ax.axhline(10.0, color='#808080', lw=0.75, ls="--")
ax.plot(xpos, E5_CASCADE, color=ss.RED, lw=ss.LINE_W, marker="o", markersize=3, label="PC-CascadeNet")
ax.plot(xpos, E5_M3, color=ss.BLUE, lw=ss.LINE_W, marker="o", markersize=3, label="Single-stage (M3)")
ax.plot(xpos, E5_GEO, color=ss.BLACK, lw=ss.LINE_W, marker="o", markersize=3, label="Geometric baseline")
ax.annotate("195.7", xy=(3, 40), xytext=(3, 36.5), ha="center", fontsize=9)
ax.text(1.05, 10.6, "Tolerance (10 mm)", fontsize=9, color='#808080')
ax.set_xticks(xpos)
ax.set_xticklabels(E5_NOISE)
ax.set_xlabel("Noise level")
ax.set_ylabel("Rebar spacing error, p90 (mm)")
ax.legend(loc="upper left")
ss.cap_axes(ax, xlim=(0.6, 4.4), ylim=(0.0, 40.0), cap_x=False)
save(fig, "fig_e5_rebar_err_p90.tiff")

# ── Fig 邻域方式对照: kNN vs 固定物理半径 ──
fig, ax = ss.new_figure()
ax.plot(NB_DENS, NB_KNN, color=ss.RED, lw=ss.LINE_W, marker="o", markersize=3,
        label="Fixed kNN (k=24)")
ax.plot(NB_DENS, NB_RADIUS, color=ss.BLUE, lw=ss.LINE_W, marker="o", markersize=3,
        label="Fixed radius (40 mm)")
ax.set_xlabel("Sampling density (pts/mm$^2$)")
ax.set_ylabel("Metal IoU (-)")
ax.legend(loc="lower right")
ss.cap_axes(ax, xlim=(0.006, 0.020), ylim=(0.0, 0.9))
save(fig, "fig_neighborhood_density.tiff")

print("[done] 全部底稿图已输出到", FIGS)
