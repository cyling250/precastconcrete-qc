"""
敏感性 S3 — 点云密度: 评估基线 cascade 在各密度子集上.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lib.utils import console, DEVICE, CHECKPOINT_BASE, CLASS_NAMES, set_seed, create_dataloaders
from lib.cascade import val_v2_simple, Stage2Model
from experiments._s1_trainer import make_s1_model


def _load_cascade(s1_arch="pointmlp"):
    """加载已训练的 baseline cascade (S1 + S2)."""
    s1_ckpt = os.path.join(CHECKPOINT_BASE, f"phase1_baseline_{s1_arch}", "best_s1.pth")
    s2_ckpt = os.path.join(CHECKPOINT_BASE, f"phase2_baseline_{s1_arch}", "best_s2.pth")

    if not os.path.exists(s1_ckpt) or not os.path.exists(s2_ckpt):
        console.print(f"[red]未找到 checkpoint: S1={s1_ckpt}, S2={s2_ckpt}[/]")
        return None, None

    s1 = make_s1_model(s1_arch).to(DEVICE)
    state = torch.load(s1_ckpt, map_location=DEVICE)
    s1.load_state_dict(state.get("model_state", state))
    s1.eval()

    s2 = Stage2Model().to(DEVICE)
    s2.load_state_dict(torch.load(s2_ckpt, map_location=DEVICE)["model_state"])
    s2.eval()
    return s1, s2


def _evaluate(s1, s2, filter_fn, desc):
    """在 filter 过滤的测试子集上评估 cascade."""
    console.print(f"\n[bold cyan]▶ {desc}[/]")
    _, _, make_test = create_dataloaders(
        8192, 16, keep_ratio=0.4, file_filter=filter_fn, only_test=True,
    )
    c_ious, c_prf, c_acc, _s, rs_iou, cm = val_v2_simple(
        s1, s2, make_test, DEVICE, CLASS_NAMES,
    )
    cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
    console.print(f"  acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}")
    return {
        "test_mIoU": c_ious["mIoU"], "test_cm": cm, "test_rs_iou": rs_iou, "test_acc": c_acc,
        **{f"iou/{k}": v for k, v in c_ious.items() if k != "mIoU"},
    }


def _print_summary(title, results):
    console.rule(f"[bold cyan]{title}[/]")
    hdr = f"{'类别':>14} | {'mIoU':>8} | {'cm':>8} | {'rsIoU':>8} | {'噪声':>8} | {'混凝土':>8} | {'钢筋':>8} | {'套筒':>8}"
    console.print(hdr)
    console.print("-" * len(hdr))
    for tag, r in results.items():
        console.print(f"{tag:>14} | {r['test_mIoU']:>8.3f} | {r['test_cm']:>8.3f} | {r['test_rs_iou']:>8.3f} | "
                      f"{r.get('iou/噪声', 0):>8.3f} | {r.get('iou/混凝土', 0):>8.3f} | "
                      f"{r.get('iou/钢筋', 0):>8.3f} | {r.get('iou/套筒', 0):>8.3f}")


def main():
    set_seed(42)
    console.rule("[bold]S3 — 点云密度敏感性[/]")

    s1, s2 = _load_cascade("pointmlp")
    if s1 is None:
        return

    levels = {
        "density_low":  (lambda f: os.path.basename(f).endswith("_low_medium.npz"),   "低密度 (0.005 pts/mm²)"),
        "density_medium": (lambda f: os.path.basename(f).endswith("_medium_medium.npz"), "中密度 (0.01 pts/mm², 基线)"),
        "density_high": (lambda f: os.path.basename(f).endswith("_high_medium.npz"),   "高密度 (0.02 pts/mm²)"),
    }

    results = {}
    for tag, (fn, desc) in levels.items():
        results[tag] = {"tag": tag, "desc": desc, **_evaluate(s1, s2, fn, desc)}

    _print_summary("S3 — 点云密度敏感性", results)
    with open(os.path.join(CHECKPOINT_BASE, "sensitivity_S3_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S3 完成![/]")


if __name__ == "__main__":
    main()
