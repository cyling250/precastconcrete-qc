# -*- coding: utf-8 -*-
"""
创建时间：2026-09-30 10:05
用途：汇总 out/ 下各补充实验 JSON 的关键数字，供逐段修改指导引用；
      同时统计稿件摘要词数（JBE 限 250 词）。
依赖：仅标准库
"""
import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(r"F:\precastconcrete-qc")


def load(rel):
    p = ROOT / rel
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def show(title, obj, keys=None, max_len=4000):
    print(f"\n===== {title} =====")
    try:
        if keys:
            for k in keys:
                print(f"{k}: {json.dumps(obj.get(k), ensure_ascii=False)[:2000]}")
        else:
            s = json.dumps(obj, ensure_ascii=False, indent=1)
            print(s[:max_len] + ("\n...(truncated)" if len(s) > max_len else ""))
    except Exception:
        traceback.print_exc()


def main():
    try:
        # 1. E1 统一口径主表
        e1 = load("out/E1/E1_tables.json")
        show("E1_tables.json", e1, max_len=6000)

        # 2. S3/S4 敏感性（修正口径 + 泄漏对照 + M3）
        for name in [
            "out/sensitivity/S3_summary.json",
            "out/sensitivity/S4_summary.json",
            "out/sensitivity/S4_summary_allkeys.json",
            "out/sensitivity/S4_summary_m3.json",
        ]:
            p = ROOT / name
            if p.exists():
                d = load(name)
                show(name, d, max_len=3000)
            else:
                print(f"\n!!!! {name} 不存在")

        # 3. E3 FPS 存活率
        show("E3/e3_fps_survival.json", load("out/E3/e3_fps_survival.json"), max_len=4000)

        # 4. E4 几何基线 val 标定 + test 报告
        show("E4/b2_val_sweep.json", load("out/E4/b2_val_sweep.json"), max_len=2000)
        show("E4/b2_test_eval.json", load("out/E4/b2_test_eval.json"), max_len=3000)

        # 5. E5 测量闭环
        show("E5/e5_summary.json", load("out/E5/e5_summary.json"), max_len=5000)
        show("E5/e5_measurement.json (截断)", load("out/E5/e5_measurement.json"), max_len=2000)

        # 6. 摘要词数与小节标题结构（需要 pandoc 提取的稿件 md，路径可传参覆盖）
        ms = ROOT / "temp_scripts" / "manuscript_extract" / "manuscript.md"
        if len(sys.argv) > 1:
            ms = Path(sys.argv[1])
        if ms.exists():
            md = ms.read_text(encoding="utf-8")
            m = re.search(r"\*\*Abstract:\*\*(.+?)\n\n", md, re.S)
            if m:
                abstract = m.group(1)
                words = re.findall(r"[A-Za-z0-9\-']+", abstract)
                print(f"\n===== Abstract 词数（按 token 计）：{len(words)} =====")
            print("\n===== 稿件标题结构 =====")
            for line in md.splitlines():
                if line.startswith("#"):
                    print(line[:100])
        else:
            print(f"\n（跳过摘要词数/标题结构：未找到稿件提取文件 {ms}，"
                  f"可 `pandoc Manuscript.docx -t markdown -o 该路径` 后重跑，或把路径作首参传入）")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
