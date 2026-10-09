"""
敏感性 S4 — 噪声水平: 评估基线 cascade 在各噪声子集上.
直接生成测试数据, 不影响 mock 目录.

噪声强度从 0.0 (低) 到 1.0 (高), 步长 0.1, 固定密度=medium.

--all-keys: 复现旧口径 (全部 70 个构型, 含训练见过的 = 泄漏), 用于量化
评测集泄漏的影响; 输出目录与 summary 文件名加 _allkeys 后缀以示区分.
"""

import argparse
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lib.utils import (
    console,
    DEVICE,
    CHECKPOINT_BASE,
    SENSITIVITY_BASE,
    CLASS_NAMES,
    set_seed,
    create_dataloaders,
)
from lib.cascade import val_v2_simple, Stage2Model
from experiments._s1_trainer import make_s1_model

from lib.pointcloud_generator._config import DENSITY_MAP, NOISE_LEVELS
from lib.pointcloud_generator._builder import BUILDERS, _filter_internal_rebar
from lib.pointcloud_generator._noise import apply_noise, apply_density_variation
from lib.pointcloud_generator._dataset import test_keys

import lib.utils.datasets as ds_module


FIXED_DENSITY = "medium"

# 只在测试划分的几何构型上评测 (修复: 原版生成全部 70 个构型,
# 其中 80% 进过训练/验证, 构成评测集泄漏)
TEST_KEYS = test_keys()

# 噪声强度: 0.0 = low, 1.0 = high, 步长 0.1
NOISE_STEP = 0.1
NOISE_INTENSITIES = [
    round(float(v), 1) for v in np.arange(0.0, 1.0 + NOISE_STEP, NOISE_STEP)
]
# => [0.0, 0.1, 0.2, ..., 1.0]

COMP_TYPES = list(BUILDERS.keys())
VARIANTS = ["rebar", "sleeve"]
N_PER_TYPE = 10


def _interpolate_noise(t: float) -> dict:
    """线性插值生成中间噪声配置. t=0=low, t=1=high."""
    low = NOISE_LEVELS["low"]
    high = NOISE_LEVELS["high"]
    cfg = {}
    for k in low:
        if isinstance(low[k], tuple):
            # 数值范围: 线性插值
            lo = np.array(low[k])
            hi = np.array(high[k])
            mid = lo + (hi - lo) * t
            if k in ("n_occlusion_polygons", "n_occlusion_sides"):
                cfg[k] = (int(round(mid[0])), int(round(mid[1])))
            else:
                cfg[k] = (float(mid[0]), float(mid[1]))
        else:
            cfg[k] = low[k] if t < 0.5 else high[k]
    return cfg


def _generate_test_data(data_dir: str, noise_t: float, seed: int = 42, all_keys: bool = False):
    """生成指定噪声强度的测试数据并保存 npz. all_keys=True 时不过滤测试划分 (旧口径)."""
    os.makedirs(data_dir, exist_ok=True)
    dens_val = DENSITY_MAP[FIXED_DENSITY]
    noise_cfg = _interpolate_noise(noise_t)
    rng_state = np.random.RandomState(seed)

    file_list = []
    n_cfg = 0
    for ct in COMP_TYPES:
        for idx in range(1, N_PER_TYPE + 1):
            if not all_keys and (ct, idx) not in TEST_KEYS:
                continue
            n_cfg += 1
            for variant in VARIANTS:
                fname = (
                    f"Mock_{ct}_{idx:03d}_{variant}_t{noise_t:.1f}_{FIXED_DENSITY}.npz"
                )
                fpath = os.path.join(data_dir, fname)
                if os.path.exists(fpath):
                    file_list.append(fpath)
                    continue

                rng = np.random.default_rng(rng_state.randint(1e9))
                builder = BUILDERS[ct]
                result = builder(density=dens_val, rng=rng, variant=variant)
                pts, lbls = _filter_internal_rebar(result["points"], result["labels"])
                pts, lbls = apply_density_variation(pts, lbls, rng)
                pts, lbls = apply_noise(pts, lbls, noise_cfg, rng)

                np.savez(fpath, points=pts, labels=lbls)
                file_list.append(fpath)

    caliber = "全部构型(旧口径)" if all_keys else "仅测试划分键"
    console.print(
        f"  [dim]生成了 {len(file_list)} 个测试样本 → {data_dir}"
        f" (评测构型 {n_cfg}/{len(COMP_TYPES) * N_PER_TYPE}, {caliber})[/]"
    )
    return data_dir


def _load_cascade(s1_arch="pointmlp"):
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


def _load_m3():
    """M3 单阶段四类模型 (A2 消融 checkpoint)."""
    from experiments._a2_trainer import SingleStageModel
    ckpt = os.path.join(CHECKPOINT_BASE, "ablation_A2", "best.pth")
    if not os.path.exists(ckpt):
        console.print(f"[red]未找到 checkpoint: {ckpt}[/]")
        return None
    model = SingleStageModel(num_classes=4, in_channels=3).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state.get("model_state", state))
    model.eval()
    return model


def _evaluate_m3(model, data_dir, desc):
    """M3 单阶段 4 类在指定数据目录上的评测 (口径与 _a2_trainer.validate_a2 一致)."""
    from experiments._a2_trainer import validate_a2
    import torch.nn as nn

    console.print(f"\n[bold cyan]▶ {desc}[/]")
    old_test_dir = ds_module.TEST_DIR
    ds_module.TEST_DIR = data_dir
    try:
        _, _, make_test = create_dataloaders(8192, 16, keep_ratio=0.4, only_test=True)
        criterion = nn.CrossEntropyLoss()
        _loss, ious, acc, prf, cm = validate_a2(model, make_test, criterion)
    finally:
        ds_module.TEST_DIR = old_test_dir
    cio = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
    console.print(
        f"  acc={acc:.3f} mIoU={ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f}"
    )
    return {
        "test_mIoU": ious["mIoU"],
        "test_cm": cm,
        "test_rs_iou": None,  # 单阶段四类无 rs 口径
        "test_acc": acc,
        **{f"iou/{k}": v for k, v in ious.items() if k != "mIoU"},
        **{f"prf/{k1}/{k2}": v2 for k1, v1 in prf.items() for k2, v2 in v1.items()},
    }


def _evaluate(s1, s2, data_dir, desc):
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
    hdr = f"{'噪声强度':>10} | {'mIoU':>8} | {'cm':>8} | {'rsIoU':>8} | {'噪声':>8} | {'混凝土':>8} | {'钢筋':>8} | {'套筒':>8}"
    console.print(hdr)
    console.print("-" * len(hdr))

    def _fmt(v):
        return f"{v:>8.3f}" if isinstance(v, (int, float)) else f"{'-':>8}"

    for tag, r in results.items():
        console.print(
            f"{tag:>10} | {_fmt(r['test_mIoU'])} | {_fmt(r['test_cm'])} | {_fmt(r['test_rs_iou'])} | "
            f"{_fmt(r.get('iou/噪声', 0))} | {_fmt(r.get('iou/混凝土', 0))} | "
            f"{_fmt(r.get('iou/钢筋', 0))} | {_fmt(r.get('iou/套筒', 0))}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-keys", action="store_true",
                    help="不过滤测试划分 (全部 70 构型, 旧口径/泄漏对照)")
    ap.add_argument("--model", choices=["cascade", "m3"], default="cascade",
                    help="评测对象: cascade=级联全管线 (默认), m3=单阶段四类 (A2)")
    args = ap.parse_args()
    # 测试数据与模型无关 (S4_noise / S4_noise_allkeys), 两类模型共用同一批数据;
    # 汇总文件按模型加 _m3 后缀区分
    data_suffix = "_allkeys" if args.all_keys else ""
    summary_suffix = data_suffix + ("_m3" if args.model == "m3" else "")

    set_seed(42)
    console.rule(f"[bold]S4 — 噪声水平敏感性 (model={args.model})[/]")
    console.print(
        f"  噪声强度: 0.0(低)~1.0(高), 步长 {NOISE_STEP}  (固定密度={FIXED_DENSITY})"
        + ("  [yellow]旧口径: 全部 70 构型 (泄漏对照)[/]" if args.all_keys else "")
    )

    if args.model == "m3":
        model = _load_m3()
        if model is None:
            return
        evaluate = lambda data_dir, desc: _evaluate_m3(model, data_dir, desc)
    else:
        s1, s2 = _load_cascade("pointmlp")
        if s1 is None:
            return
        evaluate = lambda data_dir, desc: _evaluate(s1, s2, data_dir, desc)

    base_dir = os.path.join(SENSITIVITY_BASE, f"S4_noise{data_suffix}")
    results = {}
    for t in NOISE_INTENSITIES:
        tag = f"noise_{t:.1f}"
        data_dir = os.path.join(base_dir, f"n_{t:.1f}")
        _generate_test_data(data_dir, t, all_keys=args.all_keys)
        results[tag] = {
            "noise_intensity": t,
            **evaluate(data_dir, f"噪声强度 {t:.1f}"),
        }

    # 先落盘再打印汇总, 防止打印异常丢结果
    with open(os.path.join(SENSITIVITY_BASE, f"S4_summary{summary_suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)
    _print_summary("S4 — 噪声水平敏感性", results)
    console.print(f"[bold green]S4 完成![/]")


if __name__ == "__main__":
    main()
