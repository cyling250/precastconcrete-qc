# -*- coding: utf-8 -*-
"""
B2/E4: 几何基线阈值在验证集上标定, 在测试集上报告.

协议:
  1) 网格扫描 (feat_radius_mm × lin_min) 在 mock/inuse/val 的全部 252 个文件上,
     以金属类 IoU (3 类口径, gt 钢筋+套筒合并) 为选点目标;
  2) 取 val 最优组合, 在 mock/inuse/test 的全部 252 个文件上报告完整指标
     (3 类: 噪声/混凝土/金属; 4 类: 预测金属点按该文件的配筋型式归属,
      即对几何基线最有利的口径), 并按 密度 × 噪声 × 配筋型式 汇总.

用法:
  python -m experiments.experiment_b2_geo_valtest --out out/E4
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.baselines.geometric_seg import (      # noqa: E402
    GeometricParams,
    compute_iou_np,
    segment_geometric,
)
from lib.utils import VAL_DIR, TEST_DIR, anchor_project_path  # noqa: E402


def list_npz(d: str):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz"))


def parse_name(fn: str):
    """Mock_{comp}_{vidx:03d}_{variant}_{dens}_{noise}.npz → dict (comp 可含下划线)."""
    stem = os.path.splitext(os.path.basename(fn))[0]
    p = stem.split("_")
    noise, dens, variant, vidx = p[-1], p[-2], p[-3], p[-4]
    return {
        "comp": "_".join(p[1:-4]),
        "vidx": int(vidx),
        "variant": variant,
        "dens": dens,
        "noise": noise,
    }


def eval_file(path: str, params: GeometricParams, key: str):
    """返回单文件指标: 3 类 IoU dict + 金属 IoU + (变体感知) 4 类金属 IoU."""
    d = np.load(path)
    pts, lbl = d["points"], d["labels"].astype(np.int64)
    pred = segment_geometric(pts, params=params)

    gt3 = lbl.copy()
    gt3[gt3 == 3] = 2
    iou3 = compute_iou_np(pred, gt3, {0: "噪声", 1: "混凝土", 2: "金属"})

    # 4 类 (对基线最有利): 预测金属点归入该文件实际的金属类
    metal_gt = 2 if key["variant"] == "rebar" else 3
    pred4 = pred.copy()
    pred4[pred4 == 2] = metal_gt
    iou4 = compute_iou_np(pred4, lbl, {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"})
    return iou3, iou4


def run_split(files, params, desc):
    per_file = []
    for i, f in enumerate(files):
        key = parse_name(f)
        iou3, iou4 = eval_file(f, params, key)
        per_file.append({"file": os.path.basename(f), **key,
                         "metal_iou": iou3["金属"],
                         "concrete_iou": iou3["混凝土"],
                         "noise_iou": iou3["噪声"],
                         "miou3": (iou3["噪声"] + iou3["混凝土"] + iou3["金属"]) / 3,
                         "metal_iou_4cls": iou4["钢筋" if key["variant"] == "rebar" else "套筒"]})
        if (i + 1) % 40 == 0:
            print(f"  [{desc}] {i + 1}/{len(files)}", flush=True)
    return per_file


def aggregate(per_file, keys=("variant", "dens", "noise")):
    agg = defaultdict(list)
    for r in per_file:
        agg["ALL"].append(r)
        agg[f"variant={r['variant']}"].append(r)
        agg[f"dens={r['dens']}"].append(r)
        agg[f"noise={r['noise']}"].append(r)
        for v in keys:
            pass
    out = {}
    for g, rows in agg.items():
        out[g] = {k: float(np.mean([r[k] for r in rows]))
                  for k in ("metal_iou", "concrete_iou", "noise_iou", "miou3", "metal_iou_4cls")}
        out[g]["n"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/E4")
    ap.add_argument("--radii", type=float, nargs="+", default=[30, 40, 50])
    ap.add_argument("--lins", type=float, nargs="+", default=[0.75, 0.80, 0.85])
    args = ap.parse_args()
    args.out = anchor_project_path(args.out)
    os.makedirs(args.out, exist_ok=True)

    val_files, test_files = list_npz(VAL_DIR), list_npz(TEST_DIR)
    print(f"val={len(val_files)} test={len(test_files)}", flush=True)

    # ── 1) val 网格扫描 ──
    sweep = []
    for r, lin in itertools.product(args.radii, args.lins):
        params = GeometricParams(feat_radius_mm=float(r), lin_min=float(lin))
        print(f"[sweep] radius={r} lin_min={lin}", flush=True)
        rows = run_split(val_files, params, f"r{r}-l{lin}")
        agg = aggregate(rows)
        sweep.append({"radius_mm": r, "lin_min": lin,
                      "val_metal_iou": agg["ALL"]["metal_iou"],
                      "val_miou3": agg["ALL"]["miou3"],
                      "val_rebar_metal_iou": agg["variant=rebar"]["metal_iou"],
                      "val_sleeve_metal_iou": agg["variant=sleeve"]["metal_iou"]})
        print(f"    -> val metal IoU = {sweep[-1]['val_metal_iou']:.4f}", flush=True)

    with open(os.path.join(args.out, "b2_val_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2, ensure_ascii=False)

    best = max(sweep, key=lambda s: s["val_metal_iou"])
    print(f"[best] radius={best['radius_mm']} lin_min={best['lin_min']} "
          f"(val metal IoU={best['val_metal_iou']:.4f})", flush=True)

    # ── 2) test 报告 (val 最优阈值) ──
    params = GeometricParams(feat_radius_mm=float(best["radius_mm"]),
                             lin_min=float(best["lin_min"]))
    rows = run_split(test_files, params, "test")
    agg = aggregate(rows)
    with open(os.path.join(args.out, "b2_test_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "per_file": rows, "aggregate": agg},
                  f, indent=2, ensure_ascii=False)
    print(json.dumps(agg["ALL"], indent=2))
    print(f"saved -> {args.out}/b2_val_sweep.json, b2_test_eval.json", flush=True)


if __name__ == "__main__":
    main()
