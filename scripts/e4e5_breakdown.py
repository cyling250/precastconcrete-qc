# -*- coding: utf-8 -*-
"""
创建时间：2026-09-30 10:20
用途：重算 E4 几何基线测试集均值（按 val 最优参数）与 E5 测量闭环按噪声分档的指标；
      并提取 E3 中 pointnext / pointnet++ 的逐级存活率。
依赖：仅标准库
"""
import json
import sys
import traceback
from pathlib import Path
from statistics import mean

ROOT = Path(r"F:\precastconcrete-qc")


def load(rel):
    with open(ROOT / rel, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    try:
        # ---- E4: per_file 均值（best 参数已在文件顶层给出）----
        test = load("out/E4/b2_test_eval.json")
        pf = test["per_file"]
        print(f"[E4] best 参数: r={test['best']['radius_mm']}mm lin_min={test['best']['lin_min']}, "
              f"per_file 数={len(pf)}")
        for k in ["metal_iou", "concrete_iou", "noise_iou", "miou3"]:
            vals = [f[k] for f in pf]
            print(f"  mean {k} = {mean(vals):.4f}")
        # 按噪声档拆
        for noise in ["clean", "low", "medium", "high"]:
            sub = [f for f in pf if f["noise"] == noise]
            if sub:
                print(f"  noise={noise} (n={len(sub)}): metal={mean(f['metal_iou'] for f in sub):.3f} "
                      f"concrete={mean(f['concrete_iou'] for f in sub):.3f} "
                      f"noise_iou={mean(f['noise_iou'] for f in sub):.3f}")
        # 套筒可检出性
        sleeve_files = [f for f in pf if f.get("variant") == "sleeve"]
        if sleeve_files:
            mx = max(f["metal_iou_4cls"] for f in sleeve_files)
            print(f"  sleeve 变体文件数={len(sleeve_files)}, metal_iou_4cls 最大值={mx:.4f}")

        # ---- E5: 按噪声分档 ----
        e5 = load("out/E5/e5_measurement.json")
        print(f"\n[E5] 顶层类型: {type(e5).__name__}, 键/长度: "
              f"{list(e5.keys())[:10] if isinstance(e5, dict) else len(e5)}")
        recs = e5 if isinstance(e5, list) else e5.get("records") or e5.get("per_component")
        if recs is None and isinstance(e5, dict):
            # 找第一个值是 list 的键
            for k, v in e5.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    recs = v
                    print(f"  使用键 '{k}' 作为记录表")
                    break
        if recs:
            print(f"  记录数={len(recs)}, 字段: {list(recs[0].keys())}")
            noises = sorted({r.get("noise") for r in recs})
            print(f"  噪声档: {noises}")
            for noise in noises:
                sub = [r for r in recs if r.get("noise") == noise]
                line = f"  noise={noise} (n={len(sub)}):"
                for meth in ["cascade", "m3", "geo"]:
                    errs = [abs(r[f"{meth}_long_err"]) for r in sub
                            if r.get(f"{meth}_long_valid") and r.get(f"{meth}_long_err") is not None]
                    valid = mean(1.0 if r.get(f"{meth}_long_valid") else 0.0 for r in sub)
                    if errs:
                        errs_sorted = sorted(errs)
                        p90 = errs_sorted[int(0.9 * len(errs_sorted)) - 1] if len(errs_sorted) >= 10 else errs_sorted[-1]
                        line += f" | {meth}: long p90={p90:.1f}mm valid={valid:.2f}"
                    else:
                        line += f" | {meth}: 无有效 long, valid={valid:.2f}"
                print(line)
        else:
            print("  未找到记录列表，打印前 2000 字符：")
            print(json.dumps(e5, ensure_ascii=False)[:2000])

        # ---- E3: pointnext / pointnet++ ----
        e3 = load("out/E3/e3_fps_survival.json")["result"]
        for bk in ["pointnext", "pointnet++"]:
            if bk in e3:
                runs = e3[bk]
                agg = {}
                for run in runs:
                    for stage, cls in run.items():
                        for c, v in cls.items():
                            agg.setdefault(stage, {}).setdefault(c, []).append(v)
                print(f"\n[E3] {bk}（{len(runs)} 个种子均值）:")
                for stage, cls in agg.items():
                    print(f"  {stage}: " + ", ".join(f"{c}={mean(vs):.4f}" for c, vs in cls.items()))
        # 钢筋期望存活点：220 * 累计存活率
        for bk in ["pointmlp", "pointnext"]:
            if bk in e3:
                runs = e3[bk]
                stage_names = list(runs[0].keys())
                for i, st in enumerate(stage_names):
                    surv = mean(run[st]["rebar"] for run in runs)
                    print(f"  {bk} {st}: rebar 存活率={surv:.4f}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
