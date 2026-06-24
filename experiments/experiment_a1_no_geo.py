"""
消融 A1 — 无几何特征: 输入仅 XYZ (3维), 去除 PCA 几何特征.

使用自定义 PointNet++ 变体, SA1 输入通道从 6 改为 3.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
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
    create_dataloaders,
    count_params,
    PerSampleWeightedCE,
)
from lib.pointnet2.model import SetAbstraction, FeaturePropagation

TAG = "A1"
DESC = "A1 — 无几何特征 (输入仅 XYZ, 3维)"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, f"ablation_{TAG}")


def _make_no_geo_model():
    """构建 PointNet++ 变体 (3 通道输入, 仅 XYZ)."""
    model = nn.Module()
    model.num_classes = NUM_CLASSES
    model.sa1 = SetAbstraction(512, 0.2, 32, 3, [32, 32, 64])
    model.sa2 = SetAbstraction(128, 0.4, 64, 64 + 3, [64, 64, 128])
    model.sa3 = SetAbstraction(1, None, None, 128 + 3, [128, 256, 512])
    model.fp3 = FeaturePropagation(128 + 512, [256, 256])
    model.fp2 = FeaturePropagation(64 + 256, [128, 128])
    model.fp1 = FeaturePropagation(128, [64, 32])
    model.head = nn.Sequential(
        nn.Conv1d(32, 32, 1),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Conv1d(32, NUM_CLASSES, 1),
    ).to(DEVICE)
    model.to(DEVICE)

    def forward(self, xyz, geo_features=None, scale=None):
        if scale is not None:
            scale = scale.to(xyz.device, non_blocking=True)
            r1 = 80.0 / scale
            r2 = 160.0 / scale
        else:
            r1 = r2 = None
        l1_xyz, l1_f = self.sa1(xyz, features=None, radius_override=r1)
        l2_xyz, l2_f = self.sa2(l1_xyz, l1_f, radius_override=r2)
        l3_xyz, l3_f = self.sa3(l2_xyz, l2_f)
        fp3 = self.fp3(l2_xyz, l3_xyz, l2_f, l3_f)
        fp2 = self.fp2(l1_xyz, l2_xyz, l1_f, fp3)
        fp1 = self.fp1(xyz, l1_xyz, None, fp2)
        logits = self.head(fp1)
        return logits.transpose(-1, -2)

    model.forward = forward.__get__(model)
    return model


def main():
    set_seed(42)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    console.rule(f"[bold]{DESC}[/]")

    make_train_loader, make_val_loader, make_test_loader = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4
    )

    model = _make_no_geo_model()
    console.print(f"  [dim]模型参数: {count_params(model):,}[/]")

    criterion = PerSampleWeightedCE(max_weight=15.0, noise_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

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
        console.print(f"    IoU: [{iou_str}]  mIoU={miou:.3f} | 干净mIoU={clean_miou:.3f}")

        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            best_val_ious = val_ious
            best_val_prf = val_prf
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc, "val_miou": miou,
                "val_ious": val_ious, "val_prf": val_prf,
            }, os.path.join(CHECKPOINT_DIR, f"best_{TAG}.pth"))
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
        console.print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} mIoU={test_ious['mIoU']:.3f}")

    np.savez(os.path.join(CHECKPOINT_DIR, f"history_{TAG}.npz"),
             **{k: np.array(v) for k, v in history.items()})
    with open(os.path.join(CHECKPOINT_DIR, f"history_{TAG}.json"), "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "tag": TAG, "desc": DESC, "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "best_val_noise": best_val_ious.get("噪声", 0),
        "best_val_concrete": best_val_ious.get("混凝土", 0),
        "best_val_rebar": best_val_ious.get("钢筋", 0),
        "best_val_sleeve": best_val_ious.get("套筒", 0),
        "best_val_clean_miou": (best_val_ious.get("混凝土", 0) +
                                best_val_ious.get("钢筋", 0) +
                                best_val_ious.get("套筒", 0)) / 3.0,
    }
    with open(os.path.join(CHECKPOINT_DIR, f"summary_{TAG}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[bold green]{TAG} 完成![/]")


if __name__ == "__main__":
    main()
