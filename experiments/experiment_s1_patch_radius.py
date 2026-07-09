"""
敏感性 S1 — 局部块半径: 改变 PATCH_RADIUS_MM, 其余与 phase1 baseline 一致.

测试半径: 100 / 150(基线) / 200 / 300 mm
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._s1_trainer import train_s1_radius
from lib.utils import console, CHECKPOINT_BASE, set_seed

S1_RADII = {
    "S1_100": {"radius": 100.0, "desc": "局部块半径 100mm"},
    "S1_200": {"radius": 200.0, "desc": "局部块半径 200mm"},
    "S1_300": {"radius": 300.0, "desc": "局部块半径 300mm"},
}


def main():
    set_seed(42)
    console.rule("[bold]S1 — 局部块半径敏感性[/]")

    results = {}
    for tag, cfg in S1_RADII.items():
        result = train_s1_radius(
            arch="pointmlp",
            radius=cfg["radius"],
            tag=tag,
            desc=cfg["desc"],
            num_epochs=50,
        )
        results[tag] = result

    # 汇总
    console.rule("[bold cyan]S1 — 局部块半径敏感性汇总[/]")
    header = f"{'半径':>8} | {'val_mIoU':>8} | {'噪声IoU':>8} | {'混凝土IoU':>8} | {'金属IoU':>8} | {'cm':>8}"
    console.print(header)
    console.print("-" * len(header))
    for tag, r in results.items():
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
