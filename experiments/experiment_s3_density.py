"""
敏感性 S3 — 点云密度: 评估基线 cascade 在各密度子集上.
直接生成测试数据, 不影响 mock 目录.

密度从 0.006 到 0.020, 步长 0.002, 固定噪声=medium.
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lib.utils import (
    console,
    DEVICE,
    CHECKPOINT_BASE,
    CLASS_NAMES,
    set_seed,
    create_dataloaders,
)
from lib.cascade import val_v2_simple, Stage2Model
from experiments._s1_trainer import make_s1_model

from lib.pointcloud_generator._config import NOISE_LEVELS
from lib.pointcloud_generator._builder import BUILDERS, _filter_internal_rebar
from lib.pointcloud_generator._noise import apply_noise, apply_density_variation

import lib.utils.datasets as ds_module

# 固定噪声
FIXED_NOISE = "medium"
DENSITY_STEP = 0.002
DENSITY_RANGE = [
    round(float(v), 3) for v in np.arange(0.006, 0.020 + DENSITY_STEP, DENSITY_STEP)
]
# => [0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020]

COMP_TYPES = list(BUILDERS.keys())
VARIANTS = ["rebar", "sleeve"]
N_PER_TYPE = 10


def _generate_test_data(data_dir: str, density_val: float, seed: int = 42):
    """生成指定密度等级的测试数据并保存 npz."""
    os.makedirs(data_dir, exist_ok=True)
    noise_cfg = NOISE_LEVELS[FIXED_NOISE]
    rng_state = np.random.RandomState(seed)

    file_list = []
    for ct in COMP_TYPES:
        for idx in range(1, N_PER_TYPE + 1):
            for variant in VARIANTS:
                fname = f"Mock_{ct}_{idx:03d}_{variant}_{FIXED_NOISE}_d{density_val:.3f}.npz"
                fpath = os.path.join(data_dir, fname)
                if os.path.exists(fpath):
                    file_list.append(fpath)
                    continue

                rng = np.random.default_rng(rng_state.randint(1e9))
                builder = BUILDERS[ct]
                result = builder(density=density_val, rng=rng, variant=variant)
                pts, lbls = _filter_internal_rebar(result["points"], result["labels"])
                pts, lbls = apply_density_variation(pts, lbls, rng)
                pts, lbls = apply_noise(pts, lbls, noise_cfg, rng)

                np.savez(fpath, points=pts, labels=lbls)
                file_list.append(fpath)

    console.print(f"  [dim]生成了 {len(file_list)} 个测试样本 → {data_dir}[/]")
    return data_dir


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


def _evaluate(s1, s2, data_dir, desc):
    """在生成的测试数据集上评估 cascade."""
    console.print(f"\n[bold cyan]▶ {desc}[/]")
    old_test_dir = ds_module.TEST_DIR
    ds_module.TEST_DIR = data_dir
    try:
        _, _, make_test = create_dataloaders(8192, 16, keep_ratio=0.4, only_test=True)
        c_ious, c_prf, c_acc, _s, rs_iou, cm = val_v2_simple(
            s1,
            s2,
            make_test,
            DEVICE,
            CLASS_NAMES,
        )
    finally:
        ds_module.TEST_DIR = old_test_dir
    cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
    console.print(
        f"  acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}"
    )
    return {
        "test_mIoU": c_ious["mIoU"],
        "test_cm": cm,
        "test_rs_iou": rs_iou,
        "test_acc": c_acc,
        **{f"iou/{k}": v for k, v in c_ious.items() if k != "mIoU"},
        **{f"prf/{k1}/{k2}": v2 for k1, v1 in c_prf.items() for k2, v2 in v1.items()},
    }


def _print_summary(title, results):
    console.rule(f"[bold cyan]{title}[/]")
    hdr = f"{'类别':>14} | {'mIoU':>8} | {'cm':>8} | {'rsIoU':>8} | {'噪声':>8} | {'混凝土':>8} | {'钢筋':>8} | {'套筒':>8}"
    console.print(hdr)
    console.print("-" * len(hdr))
    for tag, r in results.items():
        console.print(
            f"{tag:>14} | {r['test_mIoU']:>8.3f} | {r['test_cm']:>8.3f} | {r['test_rs_iou']:>8.3f} | "
            f"{r.get('iou/噪声', 0):>8.3f} | {r.get('iou/混凝土', 0):>8.3f} | "
            f"{r.get('iou/钢筋', 0):>8.3f} | {r.get('iou/套筒', 0):>8.3f}"
        )


def main():
    set_seed(42)
    console.rule("[bold]S3 — 点云密度敏感性[/]")
    console.print(f"  密度范围: {DENSITY_RANGE}  (固定噪声={FIXED_NOISE})")

    s1, s2 = _load_cascade("pointmlp")
    if s1 is None:
        return

    base_dir = os.path.join(CHECKPOINT_BASE, "sensitivity_S3_density")

    results = {}
    for d in DENSITY_RANGE:
        tag = f"density_{d:.3f}"
        data_dir = os.path.join(base_dir, f"d_{d:.3f}")
        _generate_test_data(data_dir, d)
        results[tag] = {
            "density": d,
            **_evaluate(s1, s2, data_dir, f"密度 {d:.3f} pts/mm²"),
        }

    _print_summary("S3 — 点云密度敏感性", results)
    with open(os.path.join(CHECKPOINT_BASE, "sensitivity_S3_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S3 完成![/]")


if __name__ == "__main__":
    main()
