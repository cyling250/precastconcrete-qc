# -*- coding: utf-8 -*-
"""
创建时间：2026-09-30 10:28
用途：解析 e5_measurement.json 嵌套记录，按方法×噪声档重算纵筋间距误差 p90、
      套筒偏位 p90 与有效率，核对《论文修改稿_补充实验.md》R5 的数字。
依赖：仅标准库
"""
import json
import sys
import traceback
from pathlib import Path
from statistics import mean

ROOT = Path(r"F:\precastconcrete-qc")


def p90(vals):
    if not vals:
        return None
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(round(0.9 * (len(s) - 1)))))
    return s[idx]


def main():
    try:
        with open(ROOT / "out/E5/e5_measurement.json", "r", encoding="utf-8") as f:
            e5 = json.load(f)
        print("tolerance_mm:", e5.get("tolerance_mm"))
        recs = e5["results"]
        print("matrix:", json.dumps(e5.get("matrix"), ensure_ascii=False))
        r0 = recs[0]
        print("\n单条记录结构示例:")
        print("  noise:", r0["noise"], "variant:", r0["variant"], "builder:", r0["builder"])
        for k in ["gt", "geo", "cascade", "m3"]:
            print(f"  {k}: {json.dumps(r0[k], ensure_ascii=False)[:400]}")

        # 逐方法×噪声聚合
        methods = ["geo", "cascade", "m3"]
        noises = ["clean", "low", "medium", "high"]
        print("\n==== 纵筋间距误差（|err| p90 / 有效率，n=每档 24 构件）====")
        for noise in noises:
            sub = [r for r in recs if r["noise"] == noise]
            parts = [f"noise={noise}"]
            for m in methods:
                errs, valid = [], []
                for r in sub:
                    d = r.get(m) or {}
                    lg = d.get("longitudinal") or {}
                    ok = lg.get("valid")
                    valid.append(1.0 if ok else 0.0)
                    if ok and lg.get("mean_spacing_err_mm") is not None:
                        errs.append(abs(lg["mean_spacing_err_mm"]))
                parts.append(f"{m}: p90={p90(errs):.1f}mm valid={mean(valid):.2f}" if errs
                             else f"{m}: 无有效 valid={mean(valid):.2f}")
            print(" | ".join(parts))

        print("\n==== 套筒偏位（p90 / 匹配数，sleeve 变体构件）====")
        for noise in noises:
            sub = [r for r in recs if r["noise"] == noise and r.get("variant") == "sleeve"]
            if not sub:
                continue
            parts = [f"noise={noise} (n={len(sub)})"]
            for m in methods:
                offs, ngt, nmatch = [], 0, 0
                for r in sub:
                    d = r.get(m) or {}
                    sl = d.get("sleeves") or {}
                    for o in sl.get("offsets_mm", []) or []:
                        offs.append(abs(o))
                    ngt += sl.get("n_gt", 0) or 0
                    nmatch += sl.get("n_matched", 0) or 0
                parts.append(f"{m}: p90={p90(offs):.1f}mm matched={nmatch}/{ngt}" if offs
                             else f"{m}: 0 匹配 ({nmatch}/{ngt})")
            print(" | ".join(parts))

        print("\n==== 发夹筋（无遮挡外露箍）端部/中部间距误差 ====")
        for noise in noises:
            sub = [r for r in recs if r["noise"] == noise and r.get("variant") == "hairpin"]
            if not sub:
                continue
            parts = [f"noise={noise} (n={len(sub)})"]
            for m in methods:
                ends = []
                for r in sub:
                    d = r.get(m) or {}
                    hp = d.get("hairpin") or {}
                    if hp.get("end_err_mm") is not None:
                        ends.append(abs(hp["end_err_mm"]))
                if ends:
                    parts.append(f"{m}: end p90={p90(ends):.1f}mm")
            print(" | ".join(parts))
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
