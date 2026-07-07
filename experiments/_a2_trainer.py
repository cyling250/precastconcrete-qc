"""
A2 消融训练: 单阶段 4 类直接分割 (噪声/混凝土/钢筋/套筒).

验证级联自适应体素降采样管线的必要性.
"""

import os, json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib.utils import (
    console, DEVICE, LR, WEIGHT_DECAY, CLASS_NAMES,
    set_seed, serialize_metrics, create_dataloaders,
    train_one_epoch, validate_one_epoch,
    build_history_template_a2, fill_history_a2, save_history,
    save_checkpoint, load_checkpoint,
)
from lib.backbones import PointMLPSeg

EPOCHS = 120


def _a2_forward_and_labels(model, batch):
    pts, feat, lbl, scale = batch
    pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
    return model(pts, scale=None), lbl


class SingleStageModel(PointMLPSeg):
    """单阶段 4 类 — 固定半径."""

    def forward(self, xyz, geo_features=None, scale=None):
        return super().forward(xyz, geo_features=None, scale=None)


def validate_a2(model, loader_fn, criterion):
    """A2 单阶段 4 类验证."""
    return validate_one_epoch(
        model, loader_fn, criterion,
        class_names=CLASS_NAMES,
        cm_fn=lambda ious: (ious["混凝土"] + ious["钢筋"] + ious["套筒"]) / 3,
        forward_and_labels_fn=_a2_forward_and_labels,
    )


def train_a2(
    num_epochs: int = EPOCHS,
    tag: str = "ablation_A2",
    desc: str = "A2 — 无级联管线 (单阶段 4 类)",
    ckpt_dir: str = None,
):
    """A2 单阶段 4 类训练主函数 (自动断点续训)."""
    from lib.utils import CHECKPOINT_BASE

    if ckpt_dir is None:
        ckpt_dir = os.path.join(CHECKPOINT_BASE, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(42)
    console.rule(f"[bold]{desc}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}")

    make_train, make_val, make_test = create_dataloaders(
        8192, 16, keep_ratio=0.4, samples_per_epoch=3000,
    )

    model = SingleStageModel(num_classes=4, in_channels=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  [dim]参数: {n_params:,}[/]")

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=num_epochs)

    # ── 断点续训 (自动检测) ──
    ckpt_path = os.path.join(ckpt_dir, "best.pth")
    start_ep = 1
    best_m = 0.0
    hist = build_history_template_a2()

    if os.path.exists(ckpt_path):
        saved_ep = load_checkpoint(ckpt_path, model, opt, sch)
        start_ep = saved_ep + 1
        hist_path = os.path.join(ckpt_dir, f"history_{tag}.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist = json.load(f)
            cm_vals = hist.get("cm", [])
            if cm_vals:
                best_m = max(cm_vals)
        console.print(f"  [green]断点续训: 从 epoch {start_ep} 继续 (最佳 mIoU={best_m:.3f})[/]")

    # ── 训练循环 ──
    for ep in range(start_ep, num_epochs + 1):
        if ep > 1:
            sch.step()

        tl = train_one_epoch(
            model, make_train(), criterion, opt, ep, tag="A2",
            forward_and_labels_fn=_a2_forward_and_labels,
        )
        vl, ious, acc, prf, cm = validate_a2(model, make_val, criterion)
        m = ious["mIoU"]

        fill_history_a2(hist, ep, tl, vl, ious, acc, prf, cm)

        io = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
        console.print(f"  A2: tl={tl:.4f} vl={vl:.4f} acc={acc:.3f} mIoU={m:.3f} | [{io}] | cm={cm:.3f}")

        if m > best_m:
            best_m = m
            save_checkpoint(ckpt_path, model, opt, sch, epoch=ep, best_miou=best_m)
            console.print(f"  [green]↑ 最佳 (mIoU={m:.3f})[/]")

        save_history(ckpt_dir, tag, hist)

    # ── 测试集 ──
    console.rule("[bold blue]测试集[/]")
    load_checkpoint(ckpt_path, model)
    tl, ious, acc, prf, cm = validate_a2(model, make_test, criterion)
    tio = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
    console.print(f"test_loss={tl:.4f} acc={acc:.3f} mIoU={ious['mIoU']:.3f} | [{tio}] | cm={cm:.3f}")

    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump({
            "best_miou": best_m, "test_miou": ious["mIoU"],
            "test_cm": cm, **serialize_metrics(ious, prf),
        }, f, indent=2)
    console.print(f"[bold green]完成! 最佳 mIoU={best_m:.3f}, 测试 mIoU={ious['mIoU']:.3f}[/]")
