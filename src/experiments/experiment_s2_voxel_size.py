"""
敏感性 S2 — S2 体素降采样大小: 改变 VOXEL_MM, 其余与 phase2 baseline 一致.

测试体素: 20~120mm, 步长 10mm (不跑 40mm), 再加 140mm
所有 checkpoint 统一放在 sensitivity_S2_voxel/ 下.
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._s2_trainer import train_s2_voxel
from lib.utils import console, CHECKPOINT_BASE, SENSITIVITY_BASE, set_seed

BASE_DIR = os.path.join(SENSITIVITY_BASE, "S2_voxel")
OLD_PREFIX = CHECKPOINT_BASE  # 历史路径: 旧版曾放在 CHECKPOINT_BASE/sensitivity_{tag}_pointmlp

VOXEL_CONFIGS = {
    f"voxel_{v}mm": {"voxel_mm": float(v), "desc": f"S2 体素 {v}mm"}
    for v in range(20, 121, 10)
    if v != 40
}
# 测试 140mm
VOXEL_CONFIGS["voxel_140mm"] = {"voxel_mm": 140.0, "desc": "S2 体素 140mm"}


# 旧的总 summary (只有基础指标, 无 per-class IoU/PRF)
_OLD_SUMMARY_PATH = os.path.join(OLD_PREFIX, "sensitivity_S2_summary.json")


def _load_old_summary() -> dict:
    """加载旧的总 summary."""
    if os.path.exists(_OLD_SUMMARY_PATH):
        return json.load(open(_OLD_SUMMARY_PATH))
    return {}


def _find_existing_summary(tag: str, arch: str = "pointmlp") -> dict | None:
    """优先新路径的 summary.json, 回退旧路径 + 旧总 summary."""
    # 1. 新路径下有 summary.json
    new_dir = os.path.join(BASE_DIR, f"{tag}_{arch}")
    new_summary = os.path.join(new_dir, "summary.json")
    if os.path.exists(new_summary):
        return json.load(open(new_summary))
    # 2. 旧路径下有 summary.json
    old_dir = os.path.join(OLD_PREFIX, f"sensitivity_{tag}_{arch}")
    old_summary = os.path.join(old_dir, "summary.json")
    if os.path.exists(old_summary):
        return json.load(open(old_summary))
    # 3. 旧路径有 best_s2.pth 但无 summary.json → 从旧总 summary 取
    if os.path.exists(os.path.join(old_dir, "best_s2.pth")):
        old_top = _load_old_summary()
        if tag in old_top:
            return old_top[tag]
    return None


def main():
    set_seed(42)
    os.makedirs(BASE_DIR, exist_ok=True)
    console.rule("[bold]S2 — S2 体素大小敏感性[/]")

    results = {}
    for tag, cfg in sorted(VOXEL_CONFIGS.items()):
        existing = _find_existing_summary(tag)
        if existing is not None:
            results[tag] = existing
            console.print(f"  [dim]已存在, 跳过: {cfg['desc']}[/]")
            continue

        new_dir = os.path.join(BASE_DIR, f"{tag}_{'pointmlp'}")
        r = train_s2_voxel(
            arch="pointmlp",
            voxel_mm=cfg["voxel_mm"],
            num_epochs=50,
            tag=tag,
            desc=cfg["desc"],
            ckpt_dir=new_dir,
        )
        if r:
            results[tag] = r

    console.rule("[bold cyan]S2 — 体素大小敏感性汇总[/]")
    header = f"{'体素':>6} | {'test_mIoU':>8} | {'噪声IoU':>8} | {'钢筋IoU':>8} | {'套筒IoU':>8} | {'cm':>6} | {'rsIoU':>6}"
    console.print(header)
    console.print("-" * len(header))
    for tag, r in sorted(results.items()):
        console.print(
            f"{r['voxel_mm']:>4.0f}mm | {r.get('test_miou', 0):>8.3f} | "
            f"{r.get('iou/噪声', 0):>8.3f} | {r.get('iou/钢筋', 0):>8.3f} | "
            f"{r.get('iou/套筒', 0):>8.3f} | {r.get('test_cm', 0):>6.3f} | {r.get('test_rs_iou', 0):>6.3f}"
        )

    with open(os.path.join(BASE_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S2 完成! (summary: {BASE_DIR}\\summary.json)[/]")


if __name__ == "__main__":
    main()
