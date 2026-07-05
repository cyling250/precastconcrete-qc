"""
敏感性 S4 — 噪声水平: 固定密度=medium, 评估基线 cascade 在各噪声子集上.

设计: 加载已训练的 phase1_baseline + phase2_baseline, 
      在噪声过滤子集上做 V2 评估.
"""

import os, json
import torch
import torch.nn.functional as F

from _common import (
    console, NUM_POINTS, BATCH_SIZE, DEVICE, CHECKPOINT_BASE, set_seed,
    create_dataloaders, CLASS_NAMES,
)
from experiment_phase1_baseline import make_s1_model
from lib.pointnet2 import Stage2Model, val_v2_simple

S1_CKPT = os.path.join(CHECKPOINT_BASE, "phase1_baseline", "best_s1.pth")
S2_CKPT = os.path.join(CHECKPOINT_BASE, "phase2_baseline", "best_s2.pth")

NOISE_LEVELS = {
    "noise_low": {
        "filter": lambda f: os.path.basename(f).endswith("_medium_low.npz"),
        "desc": "低噪声 (遮挡20% + σ=2mm + 保留85% + 噪点1%)",
    },
    "noise_medium": {
        "filter": lambda f: os.path.basename(f).endswith("_medium_medium.npz"),
        "desc": "中噪声 (遮挡30% + σ=5mm + 保留75% + 噪点3%, 基线)",
    },
    "noise_high": {
        "filter": lambda f: os.path.basename(f).endswith("_medium_high.npz"),
        "desc": "高噪声 (遮挡40% + σ=8mm + 保留65% + 噪点5%)",
    },
}


def load_baseline():
    if not os.path.exists(S1_CKPT) or not os.path.exists(S2_CKPT):
        console.print("[red]请先训练 baseline (phase1 + phase2)[/]")
        return None, None
    s1 = make_s1_model("pointnet++").to(DEVICE)
    s1.load_state_dict(torch.load(S1_CKPT, map_location=DEVICE))
    s1.eval()
    s2 = Stage2Model().to(DEVICE)
    s2.load_state_dict(torch.load(S2_CKPT, map_location=DEVICE)["model_state"])
    s2.eval()
    return s1, s2


def evaluate(s1, s2, filter_fn, desc):
    console.print(f"\n[bold cyan]▶ {desc}[/]")
    _, _, make_test = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4,
        file_filter=filter_fn, only_test=True,
    )
    c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(s1, s2, make_test, DEVICE, CLASS_NAMES)
    cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
    console.print(f"  acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}")
    return {
        "test_mIoU": c_ious["mIoU"],
        "test_cm": cm,
        "test_rs_iou": rs_iou,
        "test_acc": c_acc,
        **{f"iou/{k}": v for k, v in c_ious.items() if k != "mIoU"},
    }


def main():
    set_seed(42)
    console.rule("[bold]S4 — 噪声水平敏感性[/]")
    console.print(f"[bold]Device:[/] {DEVICE}")

    s1, s2 = load_baseline()
    if s1 is None:
        return

    results = {}
    for tag, cfg in NOISE_LEVELS.items():
        r = evaluate(s1, s2, cfg["filter"], cfg["desc"])
        results[tag] = {"tag": tag, "desc": cfg["desc"], **r}

    console.rule("[bold cyan]S4 — 噪声水平敏感性汇总[/]")
    header = f"{'噪声':>14} | {'mIoU':>8} | {'cm':>8} | {'rsIoU':>8} | {'噪声':>8} | {'混凝土':>8} | {'钢筋':>8} | {'套筒':>8}"
    console.print(header)
    console.print("-" * len(header))
    for tag, r in results.items():
        console.print(f"{tag:>14} | {r['test_mIoU']:>8.3f} | {r['test_cm']:>8.3f} | {r['test_rs_iou']:>8.3f} | "
                      f"{r.get('iou/噪声', 0):>8.3f} | {r.get('iou/混凝土', 0):>8.3f} | "
                      f"{r.get('iou/钢筋', 0):>8.3f} | {r.get('iou/套筒', 0):>8.3f}")

    with open(os.path.join(CHECKPOINT_BASE, "sensitivity_S4_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S4 完成![/]")


if __name__ == "__main__":
    main()
