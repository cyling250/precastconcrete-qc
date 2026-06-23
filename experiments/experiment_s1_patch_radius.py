"""
敏感性 S1 — 局部块半径: 改变 PATCH_RADIUS_MM, 其余与 Baseline 完全一致.

测试半径: 100 / 150(Baseline) / 200 / 300 mm
每个半径下重新训练模型, 结果保存至独立的 checkpoint 目录.
"""

import os
import json
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from lib.pointnet2 import dataset as ds_module

from _common import (
    console,
    NUM_POINTS,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    WEIGHT_DECAY,
    NUM_CLASSES,
    DEVICE,
    CLASS_NAMES,
    CHECKPOINT_BASE,
    set_seed,
    train_one_epoch,
    validate,
    _serialize_metrics,
    PointNetPlusPlus,
    create_dataloaders,
    PerSampleWeightedCE,
    count_params,
)

# S1 配置
S1_RADII = {
    "S1_100": {"radius": 100.0, "desc": "S1 — 局部块半径 100mm"},
    "S1_150": {"radius": 150.0, "desc": "S1 — 局部块半径 150mm (Baseline)"},
    "S1_200": {"radius": 200.0, "desc": "S1 — 局部块半径 200mm"},
    "S1_300": {"radius": 300.0, "desc": "S1 — 局部块半径 300mm"},
}


def run_single(tag: str, radius: float, desc: str) -> dict:
    """用给定半径训练一次模型, 返回汇总指标."""
    set_seed(42)
    checkpoint_dir = os.path.join(CHECKPOINT_BASE, f"sensitivity_{tag}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    console.rule(f"[bold]{desc}[/]")

    # 修改模块全局半径
    ds_module.PATCH_RADIUS_MM = radius

    # 数据
    make_train_loader, make_val_loader, make_test_loader = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4
    )

    # 模型
    model = PointNetPlusPlus(NUM_CLASSES).to(DEVICE)
    console.print(f"  [dim]模型参数: {count_params(model):,}[/]")

    # 损失与优化器
    criterion = PerSampleWeightedCE(max_weight=15.0, noise_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 训练循环
    best_miou = 0.0
    best_epoch = 0
    best_val_ious = {}
    best_val_prf = {}
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "clean_miou": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loader = make_train_loader()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_ious, val_acc, val_prf, clean_miou = validate(
            model, make_val_loader, criterion
        )
        scheduler.step()
        miou = val_ious["mIoU"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["clean_miou"].append(clean_miou)
        for k, v in _serialize_metrics(val_ious, val_prf).items():
            history.setdefault(k, []).append(v)

        iou_str = " | ".join(f"{k}={v:.3f}" for k, v in val_ious.items() if k != "mIoU")
        console.print(
            f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} mIoU={miou:.3f}"
        )
        console.print(
            f"    IoU: [{iou_str}]  mIoU={miou:.3f} | 干净mIoU(无噪声)={clean_miou:.3f}"
        )

        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            best_val_ious = val_ious
            best_val_prf = val_prf
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_miou": miou,
                "val_ious": val_ious,
                "val_prf": val_prf,
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, f"best_{tag}.pth"))
            console.print(f"  [green]↑ 新最佳 (mIoU={miou:.3f})[/]")

    # 测试集评估
    console.rule(f"[bold blue]{tag} 测试集评估[/]")
    ckpt_path = os.path.join(checkpoint_dir, f"best_{tag}.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state"])
        test_loss, test_ious, test_acc, test_prf, test_clean_miou = validate(
            model, make_test_loader, criterion
        )
        console.print(
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} "
            f"mIoU={test_ious['mIoU']:.3f} | 干净mIoU={test_clean_miou:.3f}"
        )
        test_metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            **_serialize_metrics(test_ious, test_prf),
        }
    else:
        test_metrics = {}

    # 保存历史与汇总
    np.savez(
        os.path.join(checkpoint_dir, f"history_{tag}.npz"),
        **{k: np.array(v) for k, v in history.items()},
    )
    with open(os.path.join(checkpoint_dir, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "tag": tag,
        "radius_mm": radius,
        "desc": desc,
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "best_val_noise": best_val_ious.get("噪声", 0),
        "best_val_concrete": best_val_ious.get("混凝土", 0),
        "best_val_rebar": best_val_ious.get("钢筋", 0),
        "best_val_clean_miou": (best_val_ious.get("混凝土", 0) + best_val_ious.get("钢筋", 0)) / 2.0,
        **test_metrics,
    }
    with open(os.path.join(checkpoint_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def print_comparison(results: dict):
    """打印 S1 各半径对比表."""
    console.rule("[bold cyan]S1 — 局部块半径敏感性汇总[/]")
    header = (
        f"{'半径':>12} | {'val_mIoU':>8} | {'噪声IoU':>8} | "
        f"{'混凝土IoU':>8} | {'钢筋IoU':>8} | {'干净mIoU':>8} | {'test_mIoU':>8}"
    )
    console.print(header)
    console.print("-" * len(header))
    for tag, r in results.items():
        console.print(
            f"{f'{r['radius_mm']:.0f}mm':>12} | {r['best_val_miou']:>8.3f} | "
            f"{r['best_val_noise']:>8.3f} | {r['best_val_concrete']:>8.3f} | "
            f"{r['best_val_rebar']:>8.3f} | {r['best_val_clean_miou']:>8.3f} | "
            f"{r.get('test_mIoU', 0):>8.3f}"
        )


def main():
    set_seed(42)
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    console.print(f"[bold]Device:[/] {DEVICE}")
    console.print("[bold]S1 — 局部块半径敏感性[/]")
    console.print("  训练 4 个模型: 半径 100 / 150(Baseline) / 200 / 300 mm\n")

    results = {}
    for tag, cfg in S1_RADII.items():
        console.print(f"\n[bold cyan]══════▶ {cfg['desc']}[/]")
        result = run_single(tag, cfg["radius"], cfg["desc"])
        results[tag] = result

    # 恢复 baseline
    ds_module.PATCH_RADIUS_MM = 150.0

    # 汇总
    print_comparison(results)
    summary_path = os.path.join(CHECKPOINT_BASE, "sensitivity_S1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold green]S1 完成! 汇总已保存至 {summary_path}[/]")


if __name__ == "__main__":
    main()
