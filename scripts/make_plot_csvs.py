# -*- coding: utf-8 -*-
"""
创建时间：2026-09-30 15:30（2026-10-08 改输出路径：JBE修改包已删除，改为 out/plot_csvs/）
用途：为稿件绘图生成全部绘图 CSV（每张图一个文件），数据取自 out/ 终版 JSON，
      B1 邻域对照为几何基线实测值（数据源已注明）。
      输出：out/plot_csvs/*.csv（utf-8-sig，Excel 可直接打开）
依赖：仅标准库
"""
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\precastconcrete-qc")
OUT = ROOT / "out" / "plot_csvs"
BACKBONES = ["pointmlp", "pointnet++", "pointnext", "pointnet"]


def load(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def write_csv(name, header, rows, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        if note:
            f.write(f"# source: {note}\n")
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  ✓ {name} ({len(rows)} rows)")


def export_history(stage):
    """Fig 7 (stage1) / Fig 8 (stage2)：四骨干训练历史逐 epoch 曲线。"""
    rows = []
    for bk in BACKBONES:
        tag = f"phase{stage}_baseline_{bk}"
        hist_path = ROOT / "out/checkpoints" / tag / f"history_{tag}.json"
        if not hist_path.exists():
            print(f"  !! 缺少 {hist_path.name}")
            continue
        h = load(f"out/checkpoints/{tag}/history_{tag}.json")
        n = len(h["epoch"])
        cm = h.get("cm") or h.get("val_cm")
        for i in range(n):
            row = [bk, h["epoch"][i], h["loss"][i], h["val_loss"][i], cm[i],
                   h.get("iou_噪声", [None] * n)[i], h.get("iou_混凝土", [None] * n)[i]]
            if stage == 1:
                row.append(h.get("iou_金属", [None] * n)[i])
            else:
                row.append(h.get("iou_钢筋", [None] * n)[i])
                row.append(h.get("iou_套筒", [None] * n)[i])
            rows.append(row)
    if stage == 1:
        header = ["backbone", "epoch", "train_loss", "val_loss", "cmIoU",
                  "iou_noise", "iou_concrete", "iou_metal(rs)"]
        note = "out/checkpoints/phase1_baseline_*/history_*.json; cm 列已是统一三类口径 cm_iou(混凝土,金属,金属), 见 src/experiments/_s1_trainer.py:69"
    else:
        header = ["backbone", "epoch", "train_loss", "val_loss", "cmIoU",
                  "iou_noise", "iou_concrete", "iou_rebar", "iou_sleeve"]
        note = "out/checkpoints/phase2_baseline_*/history_*.json; cm 取 val_cm 列(融合输出验证口径)"
    write_csv(f"fig{'7' if stage == 1 else '8'}_stage{stage}_history.csv", header, rows, note)


def export_fps():
    """Fig 9：三骨干 FPS 逐级各类点存活率（3 种子 + 均值）。"""
    e3 = load("out/E3/e3_fps_survival.json")["result"]
    rows = []
    for bk, runs in e3.items():
        stages = list(runs[0].keys())
        for st in stages:
            for cls in ["noise", "concrete", "rebar", "sleeve"]:
                vals = [run[st][cls] for run in runs]
                rows.append([bk, st, cls] + [round(v, 5) for v in vals] + [round(float(np.mean(vals)), 5)])
    write_csv("fig9_fps_survival.csv",
              ["backbone", "stage", "class", "seed1", "seed2", "seed3", "mean"],
              rows, "out/E3/e3_fps_survival.json (3,611 test patches, 3 FPS seeds)")


def export_fig10():
    """Fig 10：噪声轴 级联 rsIoU vs 几何基线 metal IoU。"""
    s4 = load("out/sensitivity/S4_summary.json")
    b1 = load("out/B1/b1_geo_v2.json")["noise_sweep"]
    geo = {}
    for r in b1:
        if r["variant"] == "rebar" and r["density"] == "medium":
            geo.setdefault(r["t"], []).append(r["rs_iou"])
    rows = []
    for k in sorted(s4.keys(), key=lambda x: float(x.split("_")[1])):
        t = s4[k]["noise_intensity"]
        rows.append([t, round(s4[k]["test_rs_iou"], 4),
                     round(float(np.mean(geo[t])), 4) if t in geo else None])
    write_csv("fig10_noise_cascade_vs_geo.csv",
              ["t", "cascade_rsIoU", "geometric_metal_IoU"],
              rows, "cascade: out/sensitivity/S4_summary.json(test_rs_iou); geo: out/B1/b1_geo_v2.json(noise_sweep, variant=rebar, density=medium, 4 构件均值)")


def export_fig11():
    """Fig 11：几何基线邻域方式对照（kNN vs 固定物理半径）。"""
    rows = [
        ["low", 0.690, 0.597],
        ["medium", 0.765, 0.306],
        ["high", 0.557, 0.116],
    ]
    write_csv("fig11_neighborhood.csv",
              ["density_level", "fixed_radius_40mm", "knn_k24"],
              rows, "《实测数据与修改意见.md》2.4(c) 实测值（金属类 IoU，rebar 变体）")


def export_fig12():
    """Fig 12：噪声轴 级联 vs 单阶段 M3（4 类 mIoU）。"""
    s4 = load("out/sensitivity/S4_summary.json")
    m3 = load("out/sensitivity/S4_summary_m3.json")
    rows = []
    for k in sorted(s4.keys(), key=lambda x: float(x.split("_")[1])):
        t = s4[k]["noise_intensity"]
        c, m = s4[k]["test_mIoU"], m3[k]["test_mIoU"]
        rows.append([t, round(c, 4), round(m, 4), round(c - m, 4)])
    write_csv("fig12_noise_cascade_vs_m3.csv",
              ["t", "cascade_mIoU", "single_stage_mIoU", "delta"],
              rows, "out/sensitivity/S4_summary.json + S4_summary_m3.json (held-out 28 样本/档)")


def export_fig13():
    """Fig 13/可选 Table 9：测量闭环按噪声档 p90。"""
    e5 = load("out/E5/e5_measurement.json")
    recs = e5["results"]
    rows = []
    for noise in ["clean", "low", "medium", "high"]:
        sub = [r for r in recs if r["noise"] == noise]
        for m in ["geo", "cascade", "m3"]:
            errs, valid = [], 0
            for r in sub:
                v = ((r.get(m) or {}).get("longitudinal") or {}).get("spacing_mm")
                if v is not None and v == v:
                    valid += 1
                    errs.append(abs(v - r["longitudinal_truth"]))
            p90_long = float(np.percentile(errs, 90)) if errs else None
            offs, ngt, nmatch = [], 0, 0
            for r in sub:
                sl = (r.get(m) or {}).get("sleeve") or {}
                ngt += sl.get("n_gt", 0) or 0
                nmatch += sl.get("n_gt_matched", 0) or 0
                for o in sl.get("offsets_mm", []) or []:
                    if o == o:
                        offs.append(abs(o))
            p90_sl = float(np.percentile(offs, 90)) if offs else None
            rows.append([noise, m, round(p90_long, 1) if p90_long is not None else None,
                         valid, len(sub),
                         round(p90_sl, 1) if p90_sl is not None else None,
                         nmatch, ngt])
    write_csv("fig13_measurement_p90.csv",
              ["noise", "method", "long_spacing_p90_mm", "long_valid_n", "long_total_n",
               "sleeve_offset_p90_mm", "sleeve_matched", "sleeve_gt_total"],
              rows, "out/E5/e5_measurement.json (96 构件 = 4 类型×3 组合×4 噪声×2 种子); method: geo/cascade/m3; 容差: 间距±10mm, 套筒±5mm")


def export_fig16():
    """Fig 16：密度鲁棒性（修正口径 + 泄漏对照列）。"""
    s3 = load("out/sensitivity/S3_summary.json")
    s3a = load("out/sensitivity/S3_summary_allkeys.json")
    rows = []
    for k in sorted(s3.keys(), key=lambda x: float(x.split("_")[1])):
        d = s3[k]
        dk = f"density_{d['density']}"
        leaked = s3a[dk]["test_mIoU"] if dk in s3a else None
        rows.append([d["density"], round(d["test_mIoU"], 4), round(d["test_cm"], 4),
                     round(d["test_rs_iou"], 4),
                     round(d["iou/噪声"], 4), round(d["iou/混凝土"], 4),
                     round(d["iou/钢筋"], 4), round(d["iou/套筒"], 4),
                     round(leaked, 4) if leaked else None])
    write_csv("fig16_density.csv",
              ["density_pts_mm2", "mIoU", "cmIoU", "rsIoU", "iou_noise", "iou_concrete",
               "iou_rebar", "iou_sleeve", "mIoU_leaked_all70"],
              rows, "out/sensitivity/S3_summary.json (held-out 28 样本/档); 末列=全部70构型旧口径对照 S3_summary_allkeys.json")


def export_fig17():
    """Fig 17：噪声鲁棒性总体三条线 + 逐类 + 泄漏对照列。"""
    s4 = load("out/sensitivity/S4_summary.json")
    s4a = load("out/sensitivity/S4_summary_allkeys.json")
    rows = []
    for k in sorted(s4.keys(), key=lambda x: float(x.split("_")[1])):
        d = s4[k]
        rows.append([d["noise_intensity"], round(d["test_mIoU"], 4), round(d["test_cm"], 4),
                     round(d["test_rs_iou"], 4),
                     round(d["iou/噪声"], 4), round(d["iou/混凝土"], 4),
                     round(d["iou/钢筋"], 4), round(d["iou/套筒"], 4),
                     round(s4a[k]["test_mIoU"], 4)])
    write_csv("fig17_noise_overall_perclass.csv",
              ["t", "mIoU", "cmIoU", "rsIoU", "iou_noise", "iou_concrete", "iou_rebar",
               "iou_sleeve", "mIoU_leaked_all70"],
              rows, "out/sensitivity/S4_summary.json (held-out 28 样本/档); 末列=全部70构型旧口径对照 S4_summary_allkeys.json")


def main():
    try:
        print("生成绘图 CSV →", OUT)
        export_history(1)
        export_history(2)
        export_fps()
        export_fig10()
        export_fig11()
        export_fig12()
        export_fig13()
        export_fig16()
        export_fig17()
        print("全部完成")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
