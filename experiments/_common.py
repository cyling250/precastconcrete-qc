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
from lib.pointnet2.metrics import (
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


# ═══════════════════════════════════════════════════
#  通用实验入口 (pluggable backbone)
# ═══════════════════════════════════════════════════


def run_experiment(
    model_factory,
    tag: str,
    desc: str = "",
    criterion: torch.nn.Module = None,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    num_epochs: int = NUM_EPOCHS,
    keep_ratio: float = KEEP_RATIO,
    batch_size: int = BATCH_SIZE,
    num_points: int = NUM_POINTS,
    seed: int = 42,
    checkpoint_subdir: str = None,
    samples_per_epoch: int = SAMPLES_PER_EPOCH,
    random_sample: bool = False,
) -> dict:
    """通用实验入口: 种子 → 数据 → 模型 → 训练 → 验证 → 测试.

    Args:
        model_factory: () -> nn.Module, 创建模型的可调用对象.
        tag: 实验标签 (用于 checkpoint 目录名).
        desc: 实验描述 (打印用).
        criterion: 损失函数, 默认 PerSampleWeightedCE(max_weight=15.0, noise_weight=0.5).
        lr / weight_decay: 优化器超参数.
        num_epochs: 训练轮数.
        keep_ratio: 采样筛选保留比例.
        batch_size / num_points: DataLoader 参数.
        seed: 随机种子.
        checkpoint_subdir: checkpoint 子目录, 默认 "checkpoints/{tag}".
        samples_per_epoch: 每 epoch 随机采样的 patch 数, None 表示全部使用.
        random_sample: 若 True, 使用纯随机采样 (A2 消融).

    Returns:
        summary dict: tag, desc, best_epoch, best_val_miou, best_val_ious,
                      test_ious, test_acc, 等.
    """

    set_seed(seed)
    ckpt_dir = (
        checkpoint_subdir
        if checkpoint_subdir
        else os.path.join(CHECKPOINT_BASE, tag)
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    if criterion is None:
        criterion = PerSampleWeightedCE(max_weight=15.0, noise_weight=0.5)

    console.rule(f"[bold]{desc}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}")

    # ── 数据 ──
    make_train_loader, make_val_loader, make_test_loader = create_dataloaders(
        num_points, batch_size, keep_ratio=keep_ratio,
        samples_per_epoch=samples_per_epoch,
        random_sample=random_sample,
    )

    # ── 模型 ──
    model = model_factory().to(DEVICE)
    console.print(f"  [dim]模型参数: {count_params(model):,}[/]")

    # ── 优化器 ──
    console.print(f"  [dim]AdamW: lr={lr}, weight_decay={weight_decay}[/]")
    console.print(f"  [dim]Criterion: {type(criterion).__name__}[/]")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ── 训练循环 ──
    best_miou = 0.0
    best_epoch = 0
    best_val_ious = {}
    best_val_prf = {}
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "clean_miou": []}

    for epoch in range(1, num_epochs + 1):
        # 现代 PyTorch: scheduler.step() 放在 epoch 开头（而非结尾）
        if epoch > 1:
            scheduler.step()

        train_loader = make_train_loader()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_ious, val_acc, val_prf, clean_miou = validate(
            model, make_val_loader, criterion
        )
        miou = val_ious["mIoU"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["clean_miou"].append(clean_miou)
        for k, v in _serialize_metrics(val_ious, val_prf).items():
            history.setdefault(k, []).append(v)

        iou_str = " | ".join(f"{k}={v:.3f}" for k, v in val_ious.items() if k != "mIoU")
        prf_str = " | ".join(
            f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
            for k, v in val_prf.items()
        )
        console.print(
            f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} mIoU={miou:.3f}"
        )
        console.print(
            f"    IoU: [{iou_str}]  mIoU={miou:.3f} | 干净mIoU(无噪声)={clean_miou:.3f}"
        )
        console.print(f"    PRF: [{prf_str}]")

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
            torch.save(ckpt, os.path.join(ckpt_dir, f"best_{tag}.pth"))
            console.print(f"  [green]↑ 新最佳 (mIoU={miou:.3f})[/]")

        # ── 每个 epoch 即时保存历史（防止中途失败丢失） ──
        _save_history(ckpt_dir, tag, history)

    # ── 测试集评估 ──
    console.rule(f"[bold blue]{tag} 测试集评估[/]")
    ckpt_path = os.path.join(ckpt_dir, f"best_{tag}.pth")
    test_metrics = {}
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
        test_iou_str = " | ".join(f"{k}={v:.3f}" for k, v in test_ious.items() if k != "mIoU")
        console.print(f"    IoU: [{test_iou_str}]")
        test_prf_str = " | ".join(
            f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
            for k, v in test_prf.items()
        )
        console.print(f"    PRF: [{test_prf_str}]")
        test_metrics = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            **_serialize_metrics(test_ious, test_prf),
        }

    # ── 保存 ──
    _save_history(ckpt_dir, tag, history)

    summary = {
        "tag": tag,
        "desc": desc,
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "best_val_noise": best_val_ious.get("噪声", 0),
        "best_val_concrete": best_val_ious.get("混凝土", 0),
        "best_val_rebar": best_val_ious.get("钢筋", 0),
        "best_val_sleeve": best_val_ious.get("套筒", 0),
        "best_val_clean_miou": (
            best_val_ious.get("混凝土", 0) +
            best_val_ious.get("钢筋", 0) +
            best_val_ious.get("套筒", 0)
        ) / 3.0,
        **test_metrics,
    }
    with open(os.path.join(ckpt_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"[bold green]{tag} 完成![/]")
    return summary


@torch.no_grad()
def evaluate_on_subset(model, criterion, file_filter: callable,
                       num_points=NUM_POINTS, batch_size=BATCH_SIZE) -> dict:
    """在文件过滤子集上评估模型, 返回指标字典."""
    _, _, make_test_loader = create_dataloaders(
        num_points, batch_size, keep_ratio=KEEP_RATIO,
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
