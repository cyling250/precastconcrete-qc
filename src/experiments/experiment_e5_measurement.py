# -*- coding: utf-8 -*-
"""
E5: 分割结果到下游测量质检的闭环实验.

阶段 1 (无需权重):
  - 按固定构造参数生成受控构件 (箍筋间距/钢筋直径已知, E5 计划书解法 A);
  - 三种"分割来源"做下游测量: ① 真值标签 (管线上界);
    ② 几何基线预测; ③ [阶段 2 接入] PC-CascadeNet / M3 预测;
  - 报告测量误差与容差合格率.

阶段 2 (需 pointmlp S1+S2 与 M3 权重, --deep 开关启用).

用法:
  python -m experiments.experiment_e5_measurement --out out/E5
  python -m experiments.experiment_e5_measurement --out out/E5 --deep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.baselines.geometric_seg import segment_geometric      # noqa: E402
from lib.measurement import (                                  # noqa: E402
    TOLERANCE_MM,
    measure_hairpin_spacing,
    measure_longitudinal_spacing,
    measure_stirrup_spacing,
    measure_sleeve_offsets,
)
from lib.pointcloud_generator._builder import BUILDERS, _filter_internal_rebar  # noqa: E402
from lib.pointcloud_generator._config import DENSITY_MAP, NOISE_LEVELS  # noqa: E402
from lib.pointcloud_generator._noise import apply_noise        # noqa: E402

# ── 受控实验矩阵 (构造参数全部已知) ──
# 4 个构件/配筋组合 × 3 组 (箍筋间距, 钢筋直径) × 4 档噪声 × 2 种子
CASES = [
    # (builder_key, variant, fixed几何, 说明)
    ("rectangular_column", "rebar",
     dict(width=600, depth=600, height=3000, n_rebars=8), "矩形柱-纵筋"),
    ("rectangular_column", "sleeve",
     dict(width=600, depth=600, height=3000, n_rebars=8), "矩形柱-套筒"),
    ("rectangular_beam", "rebar",
     dict(width=200, height=300, length=4000, n_rebars=6), "矩形梁-纵筋"),
    ("rectangular_composite_beam", "rebar",
     dict(width=200, height=300, length=4000, n_rebars=6,
          hairpin_spacing=200.0), "矩形叠合梁-纵筋+外露U形箍"),
]
COMBOS = [  # (stirrup_spacing, rebar_diam)
    (100.0, 16.0),
    (150.0, 20.0),
    (200.0, 25.0),
]
NOISES = ["clean", "low", "medium", "high"]
SEEDS = [0, 1]
DENSITY = "medium"
HAIRPIN_SPACING_TRUTH = 200.0  # 名义值; 实际按端部0.6×/中部1.4×非均匀布置


def generate(key: str, variant: str, fixed: dict, ss: float, rd: float,
             noise: str, seed: int):
    """按已知构造参数生成一个受控构件 (返回 points/labels/构造真值)."""
    rng = np.random.default_rng(1000 + seed)
    res = BUILDERS[key](**fixed, rebar_diam=rd, stirrup_spacing=ss,
                        density=DENSITY_MAP[DENSITY], rng=rng, variant=variant)
    pts, lbl = _filter_internal_rebar(res["points"], res["labels"].astype(np.int64))
    if noise != "clean":
        pts, lbl = apply_noise(pts, lbl, NOISE_LEVELS[noise], rng)
    return pts, lbl, {"stirrup_spacing": ss, "rebar_diam": rd}


def measure_component(pts: np.ndarray, lbl: np.ndarray, variant: str,
                      metal_from="gt", builder_key: str = "") -> dict:
    """对一个构件做下游测量. metal_from: 'gt'(真值标签) 或 预测标签数组.

    任务与预制构件出厂检验实际项目对应 (GB/T 51231 场景):
      - 外露纵筋间距 (rebar 变体) / 端部套筒间距 (sleeve 变体);
      - 套筒定位偏差 (sleeve 变体);
      - 叠合梁顶面外露 U 形箍筋(发夹筋)间距 (composite_beam, 端部/中部非均匀).
    柱与普通梁的侧面箍筋埋于截面内, 成品扫描不可见, 不设该任务.
    """
    pred = lbl.copy() if isinstance(metal_from, str) else np.asarray(metal_from)

    if variant == "rebar":
        metal = pts[pred == 2]
        out = {"longitudinal": measure_longitudinal_spacing(metal)}
        if "composite_beam" in builder_key:
            # 端部 trim 掉纵筋外露段后, 轴向金属点 = 发夹筋站
            out["hairpin"] = measure_hairpin_spacing(metal)
        return out
    # sleeve 变体: 金属 = 钢筋+套筒; 端部棒状簇即套筒
    metal = pts[np.isin(pred, (2, 3))]
    lon = measure_longitudinal_spacing(metal)
    sleeve_pred = pts[pred == 3]
    sleeve_gt = pts[lbl == 3]
    slv = measure_sleeve_offsets(sleeve_pred, sleeve_gt)
    return {"longitudinal": lon, "sleeve": slv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/E5")
    ap.add_argument("--deep", action="store_true",
                    help="阶段 2: 接入 PC-CascadeNet(pointmlp) 与 M3 预测")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    deep_models = (None, None, None)
    if args.deep:
        from experiments.experiment_e5_deep import load_deep_models
        deep_models = load_deep_models()  # (s1, s2, m3_model) or partially None

    results = []
    n_total = len(CASES) * len(COMBOS) * len(NOISES) * len(SEEDS)
    done = 0
    for key, variant, fixed, label in CASES:
        for ss, rd in COMBOS:
            for noise in NOISES:
                for seed in SEEDS:
                    done += 1
                    pts, lbl, truth = generate(key, variant, fixed, ss, rd, noise, seed)
                    row = {
                        "case": label, "builder": key, "variant": variant,
                        "stirrup_spacing_truth": ss, "rebar_diam": rd,
                        "noise": noise, "seed": seed, "n_points": len(pts),
                    }

                    # ① 真值上界 (验证测量管线本身); 其测量值同时作为该构件的参照真值
                    row["gt"] = measure_component(pts, lbl, variant, "gt", key)
                    row["longitudinal_truth"] = row["gt"]["longitudinal"]["spacing_mm"]

                    # ② 几何基线
                    pred_geo = segment_geometric(pts)
                    row["geo"] = measure_component(pts, lbl, variant, pred_geo, key)

                    # ③ 深度模型 (阶段 2); 只在补片覆盖的点集上测量, 与评测口径一致
                    if args.deep and deep_models[0] is not None:
                        from experiments.experiment_e5_deep import predict_cascade, predict_m3
                        pred_cas, mask_c = predict_cascade(pts, deep_models[0], deep_models[1])
                        if mask_c.any():
                            row["cascade"] = measure_component(
                                pts[mask_c], lbl[mask_c], variant, pred_cas[mask_c], key)
                        if deep_models[2] is not None:
                            pred_m3, mask_m = predict_m3(pts, deep_models[2])
                            if mask_m.any():
                                row["m3"] = measure_component(
                                    pts[mask_m], lbl[mask_m], variant, pred_m3[mask_m], key)

                    results.append(row)
                    print(f"[{done}/{n_total}] {label} ss={ss} rd={rd} "
                          f"noise={noise} seed={seed} done", flush=True)

    with open(os.path.join(args.out, "e5_measurement.json"), "w", encoding="utf-8") as f:
        json.dump({"tolerance_mm": TOLERANCE_MM, "matrix": {"cases": CASES,
                   "combos": COMBOS, "noises": NOISES, "seeds": SEEDS},
                   "results": results}, f, indent=2, ensure_ascii=False)

    # 汇总摘要 (纵筋/套筒间距误差 = 预测测量值 − 真值测量值)
    summary = defaultdict(list)
    for r in results:
        truth = r.get("longitudinal_truth")
        for src in ("gt", "geo", "cascade", "m3"):
            if src not in r:
                continue
            m = r[src]
            if "longitudinal" in m:
                v = m["longitudinal"]["spacing_mm"]
                if v == v and truth == truth:
                    summary[f"{src}__long_err"].append(v - truth)
                summary[f"{src}__long_valid"].append(1 if v == v else 0)
            if "sleeve" in m:
                summary[f"{src}__sleeve_offsets"].extend(
                    o for o in m["sleeve"]["offsets_mm"] if not np.isnan(o))
                summary[f"{src}__sleeve_n_gt"].append(m["sleeve"]["n_gt"])
                summary[f"{src}__sleeve_n_matched"].append(m["sleeve"]["n_gt_matched"])
            if "hairpin" in m:
                # 发夹筋非均匀布置: 端部真值 0.6s, 中部真值 1.4s (s=hairpin_spacing 名义值)
                s = HAIRPIN_SPACING_TRUTH
                ve, vm = m["hairpin"]["end_mm"], m["hairpin"]["mid_mm"]
                if ve == ve:
                    summary[f"{src}__hairpin_end_err"].append(ve - 0.6 * s)
                if vm == vm:
                    summary[f"{src}__hairpin_mid_err"].append(vm - 1.4 * s)
                summary[f"{src}__hairpin_valid"].append(
                    1 if (ve == ve and vm == vm) else 0)
    agg = {}
    for k, v in summary.items():
        v2 = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
        agg[k] = {"n": len(v2),
                  "mean": float(np.mean(v2)) if v2 else None,
                  "median": float(np.median(v2)) if v2 else None,
                  "p90": float(np.percentile(v2, 90)) if v2 else None}
    with open(os.path.join(args.out, "e5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"saved -> {args.out}/e5_measurement.json, e5_summary.json", flush=True)


if __name__ == "__main__":
    main()
