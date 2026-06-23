"""
敏感性 S4 — 噪声等级: 固定密度=medium, 改变噪声等级.

设计:
  1. 在完整数据集上训练一个 Baseline 模型 (或加载已有 checkpoint)
  2. 在 3 个噪声子集上分别评估: 低 / 中(Baseline) / 高
  3. 对比各噪声等级下的分割指标, 评估模型对噪声的鲁棒性 (而非重新训练)

文件命名约定: Mock_{comp}_{idx:03d}_{density}_{noise}.npz
  密度: 固定 medium
  噪声: low / medium / high

噪声等级定义 (在 medium 密度下):
  - low:     遮挡20% + σ=2mm + 保留85% + 噪点1%
  - medium:  遮挡30% + σ=5mm + 保留75% + 噪点3% (Baseline)
  - high:    遮挡40% + σ=8mm + 保留65% + 噪点5%
"""

import os
import json
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

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

TAG = "S4"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, "sensitivity_S4")
MODEL_CKPT = os.path.join(CHECKPOINT_DIR, "best_S4_noise_model.pth")

# S4 各噪声水平的文件过滤规则 (所有噪声固定 medium 密度)
NOISE_LEVELS = {
    "noise_low": {
        "filter": lambda f: os.path.basename(f).endswith("_medium_low.npz"),
        "desc": "低噪声 (遮挡20% + σ=2mm + 保留85% + 噪点1%)",
    },
    "noise_medium": {
        "filter": lambda f: os.path.basename(f).endswith("_medium_medium.npz"),
        "desc": "中噪声 (遮挡30% + σ=5mm + 保留75% + 噪点3%, Baseline)",
    },
    "noise_high": {
        "filter": lambda f: os.path.basename(f).endswith("_medium_high.npz"),
        "desc": "高噪声 (遮挡40% + σ=8mm + 保留65% + 噪点5%)",
    },
}


@torch.no_grad()
def evaluate_on_subset(model, criterion, file_filter: callable) -> dict:
    """在指定子集上评估模型, 返回指标字典."""
    _, _, make_test_loader = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4,
        file_filter=file_filter, only_test=True,
    )

    test_loss, test_ious, test_acc, test_prf, test_clean_miou = validate(
        model, make_test_loader, criterion
    )

    console.print(
        f"  test_loss={test_loss:.4f} test_acc={test_acc:.3f} "
        f"mIoU={test_ious['mIoU']:.3f} | 干净mIoU={test_clean_miou:.3f}"
    )
    iou_str = " | ".join(f"{k}={v:.3f}" for k, v in test_ious.items() if k != "mIoU")
    console.print(f"    IoU: [{iou_str}]")

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_mIoU": test_ious["mIoU"],
        "test_clean_miou": test_clean_miou,
        **_serialize_metrics(test_ious, test_prf),
    }


def train_or_load_model() -> torch.nn.Module:
    """训练 (或加载) 一个完整的 Baseline 模型."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(MODEL_CKPT):
        console.print(f"[dim]加载已有模型: {MODEL_CKPT}[/]")
        model = PointNetPlusPlus(NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_CKPT)["model_state"])
        return model

    console.print("[bold]训练 Baseline 模型 (完整数据集)...[/]")
    set_seed(42)
    make_train_loader, make_val_loader, _ = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4,
    )

    model = PointNetPlusPlus(NUM_CLASSES).to(DEVICE)
    console.print(f"  [dim]模型参数: {count_params(model):,}[/]")

    criterion = PerSampleWeightedCE(max_weight=15.0, noise_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_miou = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loader = make_train_loader()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_ious, val_acc, val_prf, clean_miou = validate(
            model, make_val_loader, criterion
        )
        scheduler.step()
        miou = val_ious["mIoU"]

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
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_miou": miou},
                MODEL_CKPT,
            )
            console.print(f"  [green]↑ 新最佳 (mIoU={miou:.3f})[/]")

    console.print(f"[green]模型训练完成, 最佳 val_mIoU={best_miou:.3f}[/]")
    return model


def print_comparison(results: dict):
    """打印 S4 各噪声水平对比表."""
    console.rule("[bold cyan]S4 — 噪声等级敏感性汇总[/]")
    header = (
        f"{'噪声':>14} | {'test_mIoU':>8} | {'噪声IoU':>8} | "
        f"{'混凝土IoU':>8} | {'钢筋IoU':>8} | {'干净mIoU':>8} | {'test_acc':>8}"
    )
    console.print(header)
    console.print("-" * len(header))
    for tag, r in results.items():
        console.print(
            f"{tag:>14} | {r.get('test_mIoU', 0):>8.3f} | "
            f"{r.get('iou/噪声', 0):>8.3f} | {r.get('iou/混凝土', 0):>8.3f} | "
            f"{r.get('iou/钢筋', 0):>8.3f} | {r.get('test_clean_miou', 0):>8.3f} | "
            f"{r.get('test_acc', 0):>8.3f}"
        )


def main():
    console.rule("[bold]S4 — 噪声等级敏感性[/]")
    console.print(f"[bold]Device:[/] {DEVICE}")
    console.print("  设计: 训练一个 Baseline 模型, 在 3 个噪声子集上评估\n")

    # 训练/加载模型
    criterion = PerSampleWeightedCE(max_weight=15.0, noise_weight=0.5)
    model = train_or_load_model()
    model.eval()

    # 在各噪声子集上评估
    results = {}
    for tag, cfg in NOISE_LEVELS.items():
        console.print(f"\n[bold cyan]▶ {cfg['desc']}[/]")
        metrics = evaluate_on_subset(model, criterion, cfg["filter"])
        results[tag] = {"tag": tag, "desc": cfg["desc"], **metrics}

    # 汇总
    print_comparison(results)
    summary_path = os.path.join(CHECKPOINT_DIR, "sensitivity_S4_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold green]S4 完成! 汇总已保存至 {summary_path}[/]")


if __name__ == "__main__":
    main()
