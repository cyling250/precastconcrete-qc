"""
共享训练模块: 训练循环、验证循环、种子设置、常量定义 (4 分类)

所有 experiment_*.py 文件从此模块导入公共组件, 避免代码重复.
"""

import os
import sys
import json
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.pointnet2 import (
    PointNetPlusPlus,
    PointNetSeg,
    PointNeXtSeg,
    PointMLPSeg,
    count_params,
    create_dataloaders,
    PerSampleWeightedCE,
)
from lib.pointnet2.metrics_utils import (
    compute_iou,
    compute_precision_recall_f1,
    compute_accuracy,
)

console = Console()

# ── 常量 (4 分类) ──
NUM_POINTS = 8192
BATCH_SIZE = 16
NUM_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 4
KEEP_RATIO = 0.4
SAMPLES_PER_EPOCH = 3000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"}
CHECKPOINT_BASE = os.path.join(os.path.dirname(__file__), "checkpoints")


def _save_history(ckpt_dir, tag, history):
    """将 history dict 保存为 .npz 和 .json（每个 epoch 调用，即时写盘防丢失）"""
    try:
        np.savez(
            os.path.join(ckpt_dir, f"history_{tag}.npz"),
            **{k: np.array(v) for k, v in history.items()},
        )
        with open(os.path.join(ckpt_dir, f"history_{tag}.json"), "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[WARN] 保存历史失败: {e}")


def set_seed(seed=42):
    """设置全局随机种子 (确保可复现)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """训练一个 epoch, 返回平均 loss."""
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
            logits = model(pts, geo_features=feat, scale=scale)
            loss = criterion(logits.permute(0, 2, 1), lbl)
            if torch.isnan(loss) or torch.isinf(loss):
                progress.update(task, advance=1, loss=float("nan"))
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
            progress.update(task, advance=1, loss=loss.item())
    return total_loss / max(valid_batches, 1)


@torch.no_grad()
def validate(model, make_val_loader, criterion):
    """验证: 返回 (avg_loss, ious, acc, prf, clean_miou)."""
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
            progress.update(
                task,
                advance=1,
                loss=loss.item()
                if not (torch.isnan(loss) or torch.isinf(loss))
                else 0.0,
            )
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    avg_loss = total_loss / len(loader)

    ious = compute_iou(preds, targets, CLASS_NAMES)
    acc = compute_accuracy(preds, targets)
    prf = compute_precision_recall_f1(preds, targets, CLASS_NAMES)
    clean_miou = (ious["混凝土"] + ious["钢筋"] + ious["套筒"]) / 3.0
    return avg_loss, ious, acc, prf, clean_miou


def _serialize_metrics(ious: dict, prf: dict) -> dict:
    """将 IoU 和 PRF 指标展平为"指标名→值"字典 (用于 JSON 保存)."""
    return {
        **{f"iou/{k}": v for k, v in ious.items()},
        **{f"prf/{k}/{m}": v for k, kv in prf.items() for m, v in kv.items()},
    }


# (run_experiment 和 evaluate_on_subset 已移除，改用 phase1/phase2 独立训练脚本)
