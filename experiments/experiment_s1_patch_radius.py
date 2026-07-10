"""
敏感性 S1 — 局部块半径: 改变 PATCH_RADIUS_MM, 其余与 phase1 baseline 一致.

测试半径: 10 mm 到 190 mm, 步长 20 mm
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._s1_trainer import train_s1_radius
from lib.utils import console, CHECKPOINT_BASE, set_seed, load_history

# 10 到 100, 步长 20 => 10, 30, 50, 70, 90
S1_RADII = {
    f"S1_{r}": {"radius": float(r), "desc": f"局部块半径 {r}mm"}
    for r in range(10, 201, 20)
}


def is_trained(ckpt_dir: str, min_epochs: int = 50) -> bool:
    """检查 checkpoint 是否已完成训练 (≥min_epochs 轮历史)."""
    hist = load_history(ckpt_dir, os.path.basename(ckpt_dir))
    if hist is None:
        hist = load_history(ckpt_dir, "S1_" + os.path.basename(ckpt_dir).split("_")[1])
    if hist and len(hist.get("epoch", [])) >= min_epochs:
        return True
    return False


def main():
    set_seed(42)
    console.rule("[bold]S1 — 局部块半径敏感性[/]")

    results = {}
    for tag, cfg in sorted(S1_RADII.items()):
        ckpt_dir = os.path.join(CHECKPOINT_BASE, f"sensitivity_{tag}_pointmlp")
        if is_trained(ckpt_dir):
            summary_path = os.path.join(ckpt_dir, "summary.json")
            if os.path.exists(summary_path):
                r = json.load(open(summary_path))
                results[tag] = r
                console.print(f"  [dim]已训练完成, 跳过: {cfg['desc']}[/]")
                continue

        result = train_s1_radius(
            arch="pointmlp",
            radius=cfg["radius"],
            tag=tag,
            desc=cfg["desc"],
            num_epochs=50,
        )
        if result:
            results[tag] = result

    # 汇总
    console.rule("[bold cyan]S1 — 局部块半径敏感性汇总[/]")
    header = f"{'半径':>8} | {'val_mIoU':>8} | {'噪声IoU':>8} | {'混凝土IoU':>8} | {'金属IoU':>8} | {'cm':>8}"
    console.print(header)
    console.print("-" * len(header))
    for tag, r in sorted(results.items()):
        console.print(
            f"{r['radius_mm']:.0f}mm | {r['best_val_miou']:>8.3f} | "
            f"{r.get('iou/噪声', 0):>8.3f} | {r.get('iou/混凝土', 0):>8.3f} | "
            f"{r.get('iou/金属', 0):>8.3f} | {r.get('test_cm', 0):>8.3f}"
        )

    with open(os.path.join(CHECKPOINT_BASE, "sensitivity_S1_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S1 完成![/]")


if __name__ == "__main__":
    main()
