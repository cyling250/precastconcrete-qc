"""
几何基线在合成测试集上的评估 (B1 实验).

产出两件事:
  1. 标准测试集 (medium 密度, low/medium/high 噪声) 上的三分类与四分类指标,
     与 PC-CascadeNet 报出的数字同口径, 可直接进对比表;
  2. 沿 11 档噪声强度轴的退化曲线, 与论文 Figure 12 的噪声鲁棒性分析同口径,
     用于定位"几何方法失效 / 深度学习方法接手"的交叉点.

噪声强度的 11 档线性插值规则与 experiments/experiment_s4_noise.py 保持一致
(t=0 -> low, t=1 -> high), 保证与主模型的曲线可叠加.

用法:
    python -m experiments.experiment_b1_geometric_baseline --out out/B1
    python -m experiments.experiment_b1_geometric_baseline --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.baselines.geometric_seg import (          # noqa: E402
    GeometricParams,
    compute_iou_np,
    rs_iou_np,
    segment_geometric,
)
from lib.pointcloud_generator._builder import (    # noqa: E402
    BUILDERS,
    _filter_internal_rebar,
)
from lib.pointcloud_generator._config import DENSITY_MAP, NOISE_LEVELS  # noqa: E402
from lib.utils import anchor_project_path                       # noqa: E402
from lib.pointcloud_generator._noise import (      # noqa: E402
    apply_density_variation,
    apply_noise,
)

N4 = {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"}
N3 = {0: "噪声", 1: "混凝土", 2: "金属"}
FIXED_DENSITY = "medium"
NOISE_STEP = 0.1
NOISE_INTENSITIES = [round(float(v), 1) for v in np.arange(0.0, 1.0 + NOISE_STEP, NOISE_STEP)]


def interpolate_noise(t: float) -> dict:
    """与 experiment_s4_noise.py 完全一致的线性插值."""
    low, high = NOISE_LEVELS["low"], NOISE_LEVELS["high"]
    cfg = {}
    for k in low:
        if isinstance(low[k], tuple):
            lo, hi = np.array(low[k]), np.array(high[k])
            mid = lo + (hi - lo) * t
            if k in ("n_occlusion_polygons", "n_occlusion_sides"):
                cfg[k] = (int(round(mid[0])), int(round(mid[1])))
            else:
                cfg[k] = (float(mid[0]), float(mid[1]))
        else:
            cfg[k] = low[k] if t < 0.5 else high[k]
    return cfg


def build(comp: str, variant: str, density: str, noise_cfg: dict, seed: int):
    """按生成器原始流程造样本 (与 mock 数据同分布), 不落盘."""
    rng = np.random.default_rng(seed)
    res = BUILDERS[comp](density=DENSITY_MAP[density], rng=rng, variant=variant)
    pts, lbls = _filter_internal_rebar(res["points"], res["labels"])
    pts, lbls = apply_density_variation(pts, lbls, rng)
    if noise_cfg is not None:
        pts, lbls = apply_noise(pts, lbls, noise_cfg, rng)
    return pts, lbls


def evaluate(pts, lbls, params: GeometricParams):
    t0 = time.time()
    pred, info = segment_geometric(pts, params, return_info=True)
    dt = time.time() - t0
    gt4 = lbls
    gt3 = lbls.copy()
    gt3[gt3 == 3] = 2
    pred3 = pred.copy()
    pred3[pred3 == 3] = 2
    i4 = compute_iou_np(pred, gt4, N4)
    i3 = compute_iou_np(pred3, gt3, N3)
    return {
        "n_points": int(len(pts)),
        "sec": round(dt, 2),
        "four": {k: round(float(v), 4) for k, v in i4.items()},
        "three": {k: round(float(v), 4) for k, v in i3.items()},
        "rs_iou": round(float(rs_iou_np(pred, gt4)), 4),
        "n_cyl_sleeve": info.get("n_cyl_sleeve", 0),
        "n_metal_pts": info.get("n_metal_pts", info.get("n_rebar_pts", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/B1")
    ap.add_argument("--types", nargs="*",
                    default=["rectangular_column", "l_shaped_column",
                             "rectangular_beam", "t_composite_beam"])
    ap.add_argument("--variants", nargs="*", default=["rebar", "sleeve"])
    ap.add_argument("--densities", nargs="*", default=["low", "medium", "high"])
    ap.add_argument("--noise-levels", nargs="*", default=["clean", "low", "medium", "high"])
    ap.add_argument("--sweep", action="store_true", help="跑 11 档噪声退化曲线")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.out = anchor_project_path(args.out)

    if args.quick:
        args.types = ["rectangular_column"]
        args.densities = ["low"]
        args.noise_levels = ["clean", "low"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    params = GeometricParams()
    results: dict = {"config": vars(args), "runs": []}

    print("=" * 100)
    print("B1 几何基线评估 (PCA 特征 + RANSAC, 三分类=Stage1 口径 / 四分类=全管线口径)")
    print("=" * 100)
    hdr = (f"{'构件':<22}{'变体':>7}{'密度':>8}{'噪声':>8}"
           f"{'金属IoU':>9}{'混凝土':>8}{'噪声IoU':>9}"
           f"{'钢筋IoU':>9}{'套筒IoU':>9}{'rsIoU':>8}{'用时s':>7}")
    print(hdr)
    print("-" * len(hdr))

    for comp in args.types:
        for variant in args.variants:
            for dens in args.densities:
                for noise in args.noise_levels:
                    cfg = None if noise == "clean" else NOISE_LEVELS[noise]
                    pts, lbls = build(comp, variant, dens, cfg, args.seed)
                    r = evaluate(pts, lbls, params)
                    r.update(comp=comp, variant=variant, density=dens, noise=noise)
                    results["runs"].append(r)
                    print(f"{comp:<22}{variant:>7}{dens:>8}{noise:>8}"
                          f"{r['three']['金属']:>9.3f}{r['three']['混凝土']:>8.3f}"
                          f"{r['three']['噪声']:>9.3f}"
                          f"{r['four']['钢筋']:>9.3f}{r['four']['套筒']:>9.3f}"
                          f"{r['rs_iou']:>8.3f}{r['sec']:>7.2f}")

    if args.sweep:
        print("\n" + "=" * 100)
        print(f"11 档噪声退化曲线 (密度固定 = {FIXED_DENSITY}, 噪声 0.0=low -> 1.0=high)")
        print("=" * 100)
        hdr2 = (f"{'构件':<22}{'变体':>7}{'t':>6}"
                f"{'金属IoU':>9}{'混凝土':>8}{'噪声IoU':>9}"
                f"{'钢筋IoU':>9}{'套筒IoU':>9}{'rsIoU':>8}")
        print(hdr2)
        print("-" * len(hdr2))
        sweep = []
        for comp in args.types:
            for variant in args.variants:
                for t in NOISE_INTENSITIES:
                    pts, lbls = build(comp, variant, FIXED_DENSITY,
                                      interpolate_noise(t), args.seed)
                    r = evaluate(pts, lbls, params)
                    r.update(comp=comp, variant=variant, density=FIXED_DENSITY, t=t)
                    sweep.append(r)
                    print(f"{comp:<22}{variant:>7}{t:>6.1f}"
                          f"{r['three']['金属']:>9.3f}{r['three']['混凝土']:>8.3f}"
                          f"{r['three']['噪声']:>9.3f}"
                          f"{r['four']['钢筋']:>9.3f}{r['four']['套筒']:>9.3f}"
                          f"{r['rs_iou']:>8.3f}")
        results["noise_sweep"] = sweep

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {args.out}.json")


if __name__ == "__main__":
    main()
