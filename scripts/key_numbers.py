# -*- coding: utf-8 -*-
"""
创建时间：2026-09-30 10:12
用途：提取逐段修改指导所需的精确数字（S4/S3/M3 曲线端点、E3 存活率、E4/E5 关键指标）。
依赖：仅标准库
"""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(r"F:\precastconcrete-qc")


def load(rel):
    with open(ROOT / rel, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    try:
        # ---- S4 修正口径：mIoU 全档 + t=1.0 逐类最低值 ----
        s4 = load("out/sensitivity/S4_summary.json")
        keys = sorted(s4.keys(), key=lambda k: float(k.split("_")[1]))
        print("[S4 corrected] t: mIoU | per-class IoU (噪声/混凝土/钢筋/套筒)")
        for k in keys:
            d = s4[k]
            print(
                f"  {d['noise_intensity']:.1f}: {d['test_mIoU']:.4f} | "
                f"{d['iou/噪声']:.3f} {d['iou/混凝土']:.3f} {d['iou/钢筋']:.3f} {d['iou/套筒']:.3f}"
            )
        last = s4[keys[-1]]
        min_class = min(
            [("噪声", last["iou/噪声"]), ("混凝土", last["iou/混凝土"]),
             ("钢筋", last["iou/钢筋"]), ("套筒", last["iou/套筒"])], key=lambda x: x[1]
        )
        print(f"  t=1.0 逐类最低: {min_class[0]} {min_class[1]:.3f}")

        # ---- S4 泄漏对照 ----
        s4a = load("out/sensitivity/S4_summary_allkeys.json")
        print("\n[S4 allkeys(泄漏)] t: mIoU | 与修正口径差值")
        for k in keys:
            d = s4a[k]
            diff = d["test_mIoU"] - s4[k]["test_mIoU"]
            print(f"  {d['noise_intensity']:.1f}: {d['test_mIoU']:.4f} | {diff:+.4f}")

        # ---- S4 M3 曲线 ----
        m3 = load("out/sensitivity/S4_summary_m3.json")
        print("\n[S4 M3] t: mIoU | cm | cascade-m3 差值")
        for k in keys:
            d = m3[k]
            diff = s4[k]["test_mIoU"] - d["test_mIoU"]
            diff_cm = s4[k]["test_cm"] - d["test_cm"]
            print(
                f"  {d['noise_intensity']:.1f}: {d['test_mIoU']:.4f} | {d['test_cm']:.4f} | "
                f"ΔmIoU {diff:+.4f} Δcm {diff_cm:+.4f}"
            )

        # ---- S3 密度轴端点 ----
        s3 = load("out/sensitivity/S3_summary.json")
        dkeys = sorted(s3.keys(), key=lambda k: float(k.split("_")[1]))
        print("\n[S3 corrected] density: mIoU")
        for k in dkeys:
            d = s3[k]
            print(f"  {d['density']:.3f}: {d['test_mIoU']:.4f}")
        s3a_path = ROOT / "out/sensitivity/S3_summary_allkeys.json"
        if s3a_path.exists():
            s3a = load("out/sensitivity/S3_summary_allkeys.json")
            ak = sorted(s3a.keys(), key=lambda k: float(k.split("_")[1]))
            print("[S3 allkeys] density: mIoU | diff")
            for k in ak:
                d = s3a[k]
                diff = d["test_mIoU"] - s3[k]["test_mIoU"]
                print(f"  {d['density']:.3f}: {d['test_mIoU']:.4f} | {diff:+.4f}")

        # ---- E3 FPS 存活率 ----
        e3 = load("out/E3/e3_fps_survival.json")
        print("\n[E3] 顶层键:", list(e3.keys()))
        for bk, v in e3.items():
            if isinstance(v, dict):
                print(f"  -- {bk}: {json.dumps(v, ensure_ascii=False)[:1200]}")

        # ---- E4 val sweep 最优 + test ----
        val = load("out/E4/b2_val_sweep.json")
        print("\n[E4 val] 类型:", type(val).__name__)
        s = json.dumps(val, ensure_ascii=False)
        print(s[:1500])
        test = load("out/E4/b2_test_eval.json")
        print("\n[E4 test]", json.dumps(test, ensure_ascii=False)[:2000])

        # ---- E5 summary ----
        e5 = load("out/E5/e5_summary.json")
        print("\n[E5 summary]", json.dumps(e5, ensure_ascii=False)[:6000])
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
