"""汇总 B1 几何基线实验结果 (读取 reports/b1_geo_v2.json)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

with open("reports/b1_geo_v2.json", encoding="utf-8") as f:
    data = json.load(f)

runs = data["runs"]
sweep = data.get("noise_sweep", [])
print(f"主表样本数={len(runs)}  噪声曲线样本数={len(sweep)}")

# ── 主表: 按 (密度, 噪声) 聚合 3 分类金属 IoU ──
print("\n" + "=" * 96)
print("A) 几何基线主表 — 按 密度 x 噪声 聚合的 3 分类指标 (均值, 括号内为样本数)")
print("=" * 96)
agg = defaultdict(lambda: defaultdict(list))
for r in runs:
    k = (r["density"], r["noise"])
    agg[k]["金属"].append(r["three"]["金属"])
    agg[k]["混凝土"].append(r["three"]["混凝土"])
    agg[k]["噪声"].append(r["three"]["噪声"])
    agg[k]["mIoU"].append(r["three"]["mIoU"])
    agg[k]["钢筋4"].append(r["four"]["钢筋"])
    agg[k]["套筒4"].append(r["four"]["套筒"])
    agg[k]["rs"].append(r["rs_iou"])
hdr = (f"{'密度':>8}{'噪声':>8}{'样本':>6}{'金属IoU':>9}{'混凝土IoU':>10}"
       f"{'噪声IoU':>9}{'mIoU':>8}{'钢筋IoU':>9}{'套筒IoU':>9}{'rsIoU':>8}")
print(hdr)
print("-" * len(hdr))
for dens in ["low", "medium", "high"]:
    for noise in ["clean", "low", "medium", "high"]:
        k = (dens, noise)
        if k not in agg:
            continue
        a = agg[k]
        print(f"{dens:>8}{noise:>8}{len(a['金属']):>6}"
              f"{np.mean(a['金属']):>9.3f}{np.mean(a['混凝土']):>10.3f}"
              f"{np.mean(a['噪声']):>9.3f}{np.mean(a['mIoU']):>8.3f}"
              f"{np.mean(a['钢筋4']):>9.3f}{np.mean(a['套筒4']):>9.3f}"
              f"{np.mean(a['rs']):>8.3f}")

# ── 按配筋型式拆开 (rebar 变体只有钢筋, sleeve 变体只有套筒) ──
print("\n" + "=" * 96)
print("B) 分变体 — 几何基线对 钢筋(rebar 变体) 与 套筒(sleeve 变体) 的检出能力")
print("=" * 96)
print(f"{'变体':>8}{'密度':>8}{'噪声':>8}{'金属IoU':>9}{'钢筋IoU':>9}{'套筒IoU':>9}{'rsIoU':>8}")
print("-" * 68)
for variant in ["rebar", "sleeve"]:
    for dens in ["low", "medium", "high"]:
        for noise in ["clean", "high"]:
            sel = [r for r in runs if r["variant"] == variant
                   and r["density"] == dens and r["noise"] == noise]
            if not sel:
                continue
            print(f"{variant:>8}{dens:>8}{noise:>8}"
                  f"{np.mean([r['three']['金属'] for r in sel]):>9.3f}"
                  f"{np.mean([r['four']['钢筋'] for r in sel]):>9.3f}"
                  f"{np.mean([r['four']['套筒'] for r in sel]):>9.3f}"
                  f"{np.mean([r['rs_iou'] for r in sel]):>8.3f}")

# ── 噪声退化曲线 ──
if sweep:
    print("\n" + "=" * 96)
    print("C) 11 档噪声退化曲线 (密度=medium) — 按变体聚合")
    print("=" * 96)
    print(f"{'变体':>8}{'t':>6}{'金属IoU':>9}{'混凝土IoU':>10}{'噪声IoU':>9}{'rsIoU':>8}")
    print("-" * 52)
    for variant in ["rebar", "sleeve"]:
        for t in sorted({r["t"] for r in sweep}):
            sel = [r for r in sweep if r["variant"] == variant and r["t"] == t]
            if not sel:
                continue
            print(f"{variant:>8}{t:>6.1f}"
                  f"{np.mean([r['three']['金属'] for r in sel]):>9.3f}"
                  f"{np.mean([r['three']['混凝土'] for r in sel]):>10.3f}"
                  f"{np.mean([r['three']['噪声'] for r in sel]):>9.3f}"
                  f"{np.mean([r['rs_iou'] for r in sel]):>8.3f}")

print("\n" + "=" * 96)
print("D) 与 PC-CascadeNet 稿件数字对比 (测试集同口径, 均值)")
print("=" * 96)
print(f"{'来源':<34}{'金属IoU':>9}{'混凝土IoU':>10}{'噪声IoU':>9}{'mIoU':>8}")
print("-" * 70)
r_sel = [r for r in runs if r["variant"] == "rebar" and r["density"] == "medium"
         and r["noise"] == "high"]
if r_sel:
    print(f"{'几何基线 rebar变体 high噪声':<34}"
          f"{np.mean([r['three']['金属'] for r in r_sel]):>9.3f}"
          f"{np.mean([r['three']['混凝土'] for r in r_sel]):>10.3f}"
          f"{np.mean([r['three']['噪声'] for r in r_sel]):>9.3f}"
          f"{np.mean([r['three']['mIoU'] for r in r_sel]):>8.3f}")
print(f"{'PC-CascadeNet (稿件 Table 6)':<34}{'0.927(rsIoU)':>9}{0.922:>10.3f}{0.906:>9.3f}{0.918:>8.3f}")
print(f"{'PointNet++ (稿件 Table 6)':<34}{'0.909(rsIoU)':>9}{0.897:>10.3f}{0.882:>9.3f}{0.896:>8.3f}")
