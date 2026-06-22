"""
Experiment: PointNet++ Baseline (局部块采样 + 几何特征 + 逐样本动态加权 CE)

训练: PatchBasedDataset — 15cm 球查询 + 钢筋比例筛选 top 40%
验证: 邻域采样 (ball query + KNN, 不做筛选) + 双缓冲并行生成

- 输入: XYZ + 40mm 邻域几何特征 (线性度/平面度/球度)
- Loss: PerSampleWeightedCE (每样本动态加权, max_weight=15.0)
- 训练: 梯度裁剪 + AMP
"""

import os, sys, json
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.pointnet2 import (
    PointNetPlusPlus,
    count_params,
    create_dataloaders,
    PerSampleWeightedCE,
)
from lib.pointnet2.metrics import (
    compute_iou,
    compute_precision_recall_f1,
    compute_accuracy,
)

console = Console()

# ── 配置 ──
NUM_POINTS = 8192
BATCH_SIZE = 8
NUM_EPOCHS = 80
LR = 5e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 3
NUM_WORKERS = 0
KEEP_RATIO = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CLASS_NAMES = {0: "噪声", 1: "混凝土", 2: "钢筋"}


# ═══════════════════════════════════════════════════
#  训练与验证
# ═══════════════════════════════════════════════════


def train_one_epoch(model, loader, criterion, optimizer, epoch, scaler):
    model.train()
    total_loss, valid_batches = 0, 0
    progress = Progress(
        TextColumn(f"[bold]Epoch {epoch:2d}[/] train"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task("train", total=len(loader), loss=0)

    with progress:
        for pts, feat, lbl, scale in loader:
            pts, feat, lbl, scale = (
                pts.to(DEVICE),
                feat.to(DEVICE),
                lbl.to(DEVICE),
                scale.to(DEVICE),
            )
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                logits = model(pts, geo_features=feat, scale=scale)
                loss = criterion(logits.permute(0, 2, 1), lbl)

            if torch.isnan(loss) or torch.isinf(loss):
                progress.update(task, advance=1, loss=float("nan"))
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            valid_batches += 1
            progress.update(task, advance=1, loss=loss.item())

    return total_loss / max(valid_batches, 1)


@torch.no_grad()
def validate(model, make_val_loader, criterion):
    """验证: 获取新的验证 DataLoader, 带进度条, 返回完整指标."""
    model.eval()
    loader = make_val_loader()
    total_loss = 0
    all_preds, all_targets = [], []

    progress = Progress(
        TextColumn("[bold]val[/]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task("val", total=len(loader), loss=0)

    with progress:
        for pts, feat, lbl, scale in loader:
            pts, feat, lbl, scale = (
                pts.to(DEVICE),
                feat.to(DEVICE),
                lbl.to(DEVICE),
                scale.to(DEVICE),
            )
            logits = model(pts, geo_features=feat, scale=scale)
            loss = criterion(logits.permute(0, 2, 1), lbl)
            if torch.isnan(loss) or torch.isinf(loss):
                total_loss += 0.0
            else:
                total_loss += loss.item()
            all_preds.append(logits.cpu())
            all_targets.append(lbl.cpu())
            progress.update(task, advance=1, loss=loss.item())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    avg_loss = total_loss / len(loader)

    ious = compute_iou(preds, targets, CLASS_NAMES)
    acc = compute_accuracy(preds, targets)
    prf = compute_precision_recall_f1(preds, targets, CLASS_NAMES)

    # 干净 mIoU (仅混凝土+钢筋, 去除噪声干扰)
    clean_miou = (ious["混凝土"] + ious["钢筋"]) / 2.0

    return avg_loss, ious, acc, prf, clean_miou


# ═══════════════════════════════════════════════════
#  日志序列化 (JSON 兼容)
# ═══════════════════════════════════════════════════


def _serialize_metrics(ious: dict, prf: dict) -> dict:
    return {
        **{f"iou/{k}": v for k, v in ious.items()},
        **{f"prf/{k}/{m}": v for k, kv in prf.items() for m, v in kv.items()},
    }


# ═══════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════


def main():
    console.print(f"[bold]Device:[/] {DEVICE}")
    console.print(
        f"[bold]设置:[/] {NUM_POINTS}点, batch={BATCH_SIZE}, {NUM_EPOCHS}epochs, "
        f"keep_ratio={KEEP_RATIO}"
    )

    # 数据 (双缓冲: 训练 + 验证 + 测试 并行生成)
    make_train_loader, make_val_loader, make_test_loader = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=KEEP_RATIO
    )

    # 模型
    model = PointNetPlusPlus(NUM_CLASSES).to(DEVICE)
    console.print(f"[bold]模型参数:[/] {count_params(model):,}")

    criterion = PerSampleWeightedCE(max_weight=15.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    # ── 训练循环 ──
    best_miou = 0.0
    best_epoch = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "clean_miou": [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loader = make_train_loader()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, scaler
        )
        val_loss, val_ious, val_acc, val_prf, clean_miou = validate(
            model, make_val_loader, criterion
        )
        scheduler.step()

        miou = val_ious["mIoU"]

        # 保存历史
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["clean_miou"].append(clean_miou)
        for k, v in _serialize_metrics(val_ious, val_prf).items():
            history.setdefault(k, []).append(v)

        # 日志
        iou_str = " | ".join(f"{k}={v:.3f}" for k, v in val_ious.items() if k != "mIoU")
        prf_str = " | ".join(
            f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
            for k, v in val_prf.items()
        )
        console.print(
            f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} mIoU={miou:.3f}"
        )
        console.print(f"    IoU: [{iou_str}]  mIoU={miou:.3f} | 干净mIoU(无噪声)={clean_miou:.3f}")
        console.print(f"    PRF: [{prf_str}]")

        # 保存最佳 (用 mIoU)
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_miou": miou,
                "val_ious": val_ious,
                "val_prf": val_prf,
            }
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best_pointnet2.pth"))
            console.print(f"  ✓ 保存最佳模型 (mIoU={miou:.3f})")

    # ── 最终报告 ──
    console.rule("[bold green]训练完成")
    console.print(f"[bold]最佳 epoch:[/] {best_epoch}  (mIoU={best_miou:.3f})")

    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_pointnet2.pth"))
    console.print(f"[bold]最佳模型 (epoch={ckpt['epoch']}):[/]")
    for k, v in ckpt["val_ious"].items():
        console.print(f"  {k}: {v:.3f}")
    if "val_prf" in ckpt:
        console.print("[bold]最佳模型 PRF:[/]")
        for name, m in ckpt["val_prf"].items():
            console.print(
                f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
            )

    # ── 测试集评估 ──
    console.rule("[bold blue]测试集评估")
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_ious, test_acc, test_prf, test_clean_miou = validate(
        model, make_test_loader, criterion
    )
    console.print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} "
                  f"mIoU={test_ious['mIoU']:.3f} | 干净mIoU={test_clean_miou:.3f}")
    iou_str = " | ".join(f"{k}={v:.3f}" for k, v in test_ious.items() if k != "mIoU")
    console.print(f"    IoU: [{iou_str}]  mIoU={test_ious['mIoU']:.3f}")
    prf_str = " | ".join(
        f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
        for k, v in test_prf.items()
    )
    console.print(f"    PRF: [{prf_str}]")

    # 保存历史
    np.savez(
        os.path.join(CHECKPOINT_DIR, "history.npz"),
        **{k: np.array(v) for k, v in history.items()},
    )
    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, "best_metrics.json"), "w") as f:
        json.dump(
            {
                "epoch": ckpt["epoch"],
                "val_acc": ckpt["val_acc"],
                "val_miou": ckpt["val_miou"],
                **_serialize_metrics(ckpt["val_ious"], ckpt.get("val_prf", {})),
            },
            f,
            indent=2,
        )

    console.print("[bold green]Done![/]")


if __name__ == "__main__":
    main()
