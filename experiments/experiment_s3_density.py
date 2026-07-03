"""
敏感性 S3 — 点云密度: 固定噪声=medium, 评估基线 cascade 在各密度子集上.

设计: 加载已训练的 phase1_baseline + phase2_baseline, 
      在密度过滤子集上做 V2 评估.
"""

import os, json
import torch
import torch.nn.functional as F

from _common import (
    console, NUM_POINTS, BATCH_SIZE, DEVICE, CHECKPOINT_BASE, set_seed,
    create_dataloaders, CLASS_NAMES,
)
from experiment_phase1_baseline import make_s1_model
from experiment_phase2_baseline import Stage2Model, val_v2_simple

S1_CKPT = os.path.join(CHECKPOINT_BASE, "phase1_baseline", "best_s1.pth")
S2_CKPT = os.path.join(CHECKPOINT_BASE, "phase2_baseline", "best_s2.pth")

DENSITY_LEVELS = {
    "density_low": {
        "filter": lambda f: os.path.basename(f).endswith("_low_medium.npz"),
        "desc": "低密度 (0.005 pts/mm²)",
    },
    "density_medium": {
        "filter": lambda f: os.path.basename(f).endswith("_medium_medium.npz"),
        "desc": "中密度 (0.01 pts/mm², 基线)",
    },
    "density_high": {
        "filter": lambda f: os.path.basename(f).endswith("_high_medium.npz"),
        "desc": "高密度 (0.02 pts/mm²)",
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
    c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(s1, s2, make_test)
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
    console.rule("[bold]S3 — 点云密度敏感性[/]")
    console.print(f"[bold]Device:[/] {DEVICE}")

    s1, s2 = load_baseline()
    if s1 is None:
        return

    results = {}
    for tag, cfg in DENSITY_LEVELS.items():
        r = evaluate(s1, s2, cfg["filter"], cfg["desc"])
        results[tag] = {"tag": tag, "desc": cfg["desc"], **r}

    # 汇总
    console.rule("[bold cyan]S3 — 点云密度敏感性汇总[/]")
    header = f"{'密度':>14} | {'mIoU':>8} | {'cm':>8} | {'rsIoU':>8} | {'噪声':>8} | {'混凝土':>8} | {'钢筋':>8} | {'套筒':>8}"
    console.print(header)
    console.print("-" * len(header))
    for tag, r in results.items():
        console.print(f"{tag:>14} | {r['test_mIoU']:>8.3f} | {r['test_cm']:>8.3f} | {r['test_rs_iou']:>8.3f} | "
                      f"{r.get('iou/噪声', 0):>8.3f} | {r.get('iou/混凝土', 0):>8.3f} | "
                      f"{r.get('iou/钢筋', 0):>8.3f} | {r.get('iou/套筒', 0):>8.3f}")

    with open(os.path.join(CHECKPOINT_BASE, "sensitivity_S3_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S3 完成![/]")


if __name__ == "__main__":
    main()
