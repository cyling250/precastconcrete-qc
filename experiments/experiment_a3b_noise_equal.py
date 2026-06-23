"""
消融 A3b — 噪声感知 α=1.0 (平等对待噪声类).

PerSampleWeightedCE 中 noise_weight=1.0, 噪声与其他类同等加权.
"""

import os
import json
import numpy as np
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

TAG = "A3b"
DESC = "A3b — 噪声感知 α=1.0 (平等对待噪声类)"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, f"ablation_{TAG}")


def main():
    set_seed(42)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    console.rule(f"[bold]{DESC}[/]")

    # 数据
    make_train_loader, make_val_loader, make_test_loader = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4
    )

    # 模型
    model = PointNetPlusPlus(NUM_CLASSES).to(DEVICE)
    console.print(f"  [dim]模型参数: {count_params(model):,}[/]")

    # 损失: noise_weight=1.0
    criterion = PerSampleWeightedCE(max_weight=15.0, noise_weight=1.0)
    console.print(f"  [dim]Loss: PerSampleWeightedCE (noise_weight=1.0)[/]")

    # 优化器
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
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"best_{TAG}.pth"))
            console.print(f"  [green]↑ 新最佳 (mIoU={miou:.3f})[/]")

    # 测试集评估
    console.rule(f"[bold blue]{TAG} 测试集评估[/]")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{TAG}.pth")
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

    # 保存历史与汇总
    np.savez(
        os.path.join(CHECKPOINT_DIR, f"history_{TAG}.npz"),
        **{k: np.array(v) for k, v in history.items()},
    )
    with open(os.path.join(CHECKPOINT_DIR, f"history_{TAG}.json"), "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "tag": TAG,
        "desc": DESC,
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "best_val_noise": best_val_ious.get("噪声", 0),
        "best_val_concrete": best_val_ious.get("混凝土", 0),
        "best_val_rebar": best_val_ious.get("钢筋", 0),
        "best_val_clean_miou": (best_val_ious.get("混凝土", 0) + best_val_ious.get("钢筋", 0)) / 2.0,
    }
    with open(os.path.join(CHECKPOINT_DIR, f"summary_{TAG}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[bold green]{TAG} 完成![/]")


if __name__ == "__main__":
    main()
