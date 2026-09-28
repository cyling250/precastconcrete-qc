"""
诊断: 按真值类别统计 PCA 局部几何特征, 用于确定几何基线的候选集阈值.

用法: python -m lib.baselines.diag_features
"""

from __future__ import annotations

import argparse

import numpy as np

from lib.pointcloud_generator._builder import BUILDERS, _filter_internal_rebar
from lib.pointcloud_generator._config import DENSITY_MAP, NOISE_LEVELS
from lib.pointcloud_generator._noise import apply_density_variation, apply_noise

from .geometric_seg import local_pca_features

NAMES = {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"}
FEATS = ["linearity", "planarity", "sphericity", "curvature", "knn_dist"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="rectangular_column")
    ap.add_argument("--variant", default="rebar")
    ap.add_argument("--density", default="low")
    ap.add_argument("--noise", default="clean")
    ap.add_argument("--knn", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    res = BUILDERS[args.type](density=DENSITY_MAP[args.density], rng=rng, variant=args.variant)
    pts, lbls = _filter_internal_rebar(res["points"], res["labels"])
    pts, lbls = apply_density_variation(pts, lbls, rng)
    if args.noise != "clean":
        pts, lbls = apply_noise(pts, lbls, NOISE_LEVELS[args.noise], rng)

    feat = local_pca_features(pts, args.knn)
    print(f"\n[{args.type}/{args.variant}/dens={args.density}/noise={args.noise}] N={len(pts):,}")
    print("每类点数: " + ", ".join(f"{NAMES[c]}={int((lbls == c).sum())}" for c in range(4)))
    print(f"\n{'特征':<12} {'类别':<8} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8}")
    print("-" * 68)
    for f in FEATS:
        v = feat[f]
        for c in range(4):
            m = lbls == c
            if m.sum() < 5:
                continue
            q = np.percentile(v[m], [10, 25, 50, 75, 90])
            print(f"{f:<12} {NAMES[c]:<8} " + " ".join(f"{x:>8.4f}" for x in q))
        print()

    # 候选集覆盖率: 不同曲率阈值能覆盖多少钢筋/套筒点, 同时引入多少混凝土点
    print("候选集筛选 (curvature 阈值) — 召回率 / 误纳率:")
    cur = feat["curvature"]
    tot_r = max(1, int(((lbls == 2) | (lbls == 3)).sum()))
    tot_c = max(1, int((lbls == 1).sum()))
    for t in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        m = cur >= t
        rec = int(((lbls == 2) | (lbls == 3))[m].sum()) / tot_r
        inc = int((lbls == 1)[m].sum()) / tot_c
        print(f"  cur>={t:<6} 钢筋+套筒召回={rec:6.2%}   混凝土误纳={inc:6.2%}   "
              f"候选点数={int(m.sum()):,}")

    print("\n候选集筛选 (linearity 阈值):")
    lin = feat["linearity"]
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        m = lin >= t
        rec = int(((lbls == 2) | (lbls == 3))[m].sum()) / tot_r
        inc = int((lbls == 1)[m].sum()) / tot_c
        print(f"  lin>={t:<6} 钢筋+套筒召回={rec:6.2%}   混凝土误纳={inc:6.2%}   "
              f"候选点数={int(m.sum()):,}")


if __name__ == "__main__":
    main()
