"""扫描固定物理邻域半径与线性度阈值, 找跨密度可复用的工作点."""

from __future__ import annotations

import numpy as np

from lib.pointcloud_generator._builder import BUILDERS, _filter_internal_rebar
from lib.pointcloud_generator._config import DENSITY_MAP, NOISE_LEVELS
from lib.pointcloud_generator._noise import apply_density_variation, apply_noise

from lib.baselines.geometric_seg import (
    GeometricParams,
    compute_iou_np,
    local_pca_features,
    segment_geometric,
)

N3 = {0: "噪声", 1: "混凝土", 2: "金属"}
COMPS = ["rectangular_column", "rectangular_beam"]
VARIANTS = ["rebar", "sleeve"]
DENS = ["low", "medium", "high"]


def build(comp, variant, dens, noise):
    rng = np.random.default_rng(42)
    res = BUILDERS[comp](density=DENSITY_MAP[dens], rng=rng, variant=variant)
    pts, lbls = _filter_internal_rebar(res["points"], res["labels"])
    pts, lbls = apply_density_variation(pts, lbls, rng)
    if noise != "clean":
        pts, lbls = apply_noise(pts, lbls, NOISE_LEVELS[noise], rng)
    gt = lbls.copy()
    gt[gt == 3] = 2
    return pts, gt


print("A) 固定半径邻域下的线性度分布 (真值金属 vs 混凝土, clean 数据)")
print(f"{'半径mm':>8}{'密度':>9}{'变体':>8}{'金属P25':>10}{'金属P50':>10}"
      f"{'混凝土P75':>11}{'混凝土P90':>11}{'分离阈值':>10}")
print("-" * 78)
cache = {}
for rad in [30.0, 40.0, 60.0]:
    for dens in DENS:
        for variant in VARIANTS:
            pts, gt = build(COMPS[0], variant, dens, "clean")
            if len(pts) > 200_000:
                continue
            f = local_pca_features(pts, 24, radius_mm=rad)
            lin = f["linearity"]
            m = gt == 2
            c = gt == 1
            if m.sum() < 10:
                continue
            mp25, mp50 = np.percentile(lin[m], [25, 50])
            cp75, cp90 = np.percentile(lin[c], [75, 90])
            thr = 0.5 * (mp25 + cp90)
            print(f"{rad:>8.0f}{dens:>9}{variant:>8}{mp25:>10.3f}{mp50:>10.3f}"
                  f"{cp75:>11.3f}{cp90:>11.3f}{thr:>10.3f}")

print("\nB) 端到端 3 分类金属 IoU (按 半径 x lin_min 扫描)")
print(f"{'半径mm':>8}{'lin_min':>9}", end="")
for dens in DENS:
    for variant in VARIANTS:
        print(f"{dens[:3]+'/'+variant[:3]:>12}", end="")
print()
print("-" * (17 + 12 * len(DENS) * len(VARIANTS)))
best = None
for rad in [30.0, 40.0, 60.0]:
    for lin_min in [0.60, 0.70, 0.80, 0.90]:
        row = []
        for dens in DENS:
            for variant in VARIANTS:
                pts, gt = build(COMPS[0], variant, dens, "clean")
                p = GeometricParams(feat_radius_mm=rad, lin_min=lin_min, three_class=True)
                pred, _ = segment_geometric(pts, p, return_info=True)
                iou = compute_iou_np(pred, gt, N3)["金属"]
                row.append(iou)
        print(f"{rad:>8.0f}{lin_min:>9.2f}" + "".join(f"{v:>12.3f}" for v in row))
        score = float(np.mean(row))
        if best is None or score > best[0]:
            best = (score, rad, lin_min)
print(f"\n最优 (按各格平均): 半径={best[1]:.0f}mm  lin_min={best[2]:.2f}  平均金属IoU={best[0]:.3f}")
