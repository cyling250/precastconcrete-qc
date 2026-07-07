"""
敏感性 S2 — S2 体素降采样大小: 改变 VOXEL_MM, 其余与 phase2 baseline 一致.

测试体素: 20mm / 40mm(基线) / 80mm
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._s2_trainer import train_s2_voxel
from lib.utils import console, CHECKPOINT_BASE, set_seed

VOXEL_CONFIGS = {
    "voxel_20mm": {"voxel_mm": 20.0, "desc": "S2 体素 20mm"},
    "voxel_40mm": {"voxel_mm": 40.0, "desc": "S2 体素 40mm (基线)"},
    "voxel_80mm": {"voxel_mm": 80.0, "desc": "S2 体素 80mm"},
}


def main():
    set_seed(42)
    console.rule("[bold]S2 — S2 体素大小敏感性[/]")

    results = {}
    for tag, cfg in VOXEL_CONFIGS.items():
        r = train_s2_voxel(
            arch="pointmlp", voxel_mm=cfg["voxel_mm"],
            tag=tag, desc=cfg["desc"],
        )
        if r:
            results[tag] = r

    console.rule("[bold cyan]S2 — 体素大小敏感性汇总[/]")
    for tag, r in results.items():
        console.print(f"  {tag}: {r['voxel_mm']:.0f}mm | test_mIoU={r['test_miou']:.3f} cm={r['test_cm']:.3f} rsIoU={r['test_rs_iou']:.3f}")

    with open(os.path.join(CHECKPOINT_BASE, "sensitivity_S2_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S2 完成![/]")


if __name__ == "__main__":
    main()
