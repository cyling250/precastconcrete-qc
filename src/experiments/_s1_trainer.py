"""
S1 训练器: 3类分割 (噪声/混凝土/金属).

支持:
  - 多架构 (pointnet++/pointnet/pointmlp/pointnext)
  - 断点续训 (resume)
  - 早停
  - 完整指标记录
"""

import os, json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib.utils import (
    console,
    DEVICE,
    LR,
    WEIGHT_DECAY,
    S1_NAMES,
    set_seed,
    map_labels,
    serialize_metrics,
    create_dataloaders,
    train_one_epoch,
    validate_one_epoch,
    build_history_template_s1,
    fill_history_s1,
    save_history,
    save_checkpoint,
    load_checkpoint,
    cm_iou,
)
from experiments._config import get_backbone_cls


def _s1_forward_and_labels(model, batch):
    """S1 3类: XYZ → logits, 标签 4→3 映射."""
    pts, feat, lbl, scale = batch
    pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
    return model(pts, scale=None), map_labels(lbl)


class S1Model(nn.Module):
    """S1 3分类统一包装器: 固定半径, 仅XYZ, 可替换骨干."""

    def __init__(self, arch: str = "pointmlp"):
        super().__init__()
        self.arch = arch
        self.backbone = get_backbone_cls(arch)(num_classes=3, in_channels=3)

    def forward(self, xyz, geo_features=None, scale=None):
        return self.backbone(xyz, geo_features=None, scale=None)


def make_s1_model(arch: str = "pointmlp") -> S1Model:
    return S1Model(arch)


def validate_s1(model, loader_fn, criterion):
    """S1 3类验证."""
    return validate_one_epoch(
        model,
        loader_fn,
        criterion,
        class_names=S1_NAMES,
        # Stage 1 的 cmIoU 与 Eq.(9) 统一: 金属类同时代表钢筋与套筒两个位置
        cm_fn=lambda ious: cm_iou(ious["混凝土"], ious["金属"], ious["金属"]),
        forward_and_labels_fn=_s1_forward_and_labels,
    )


def train_s1(
    arch: str = "pointmlp",
    num_epochs: int = 80,
    early_stop_patience: int = 10,
    tag: str = "phase1_baseline",
    desc: str = "S1",
    ckpt_dir: str = None,
    dataloader_kwargs: dict = None,
    val_batch_mult: int = 1,
    batch_size: int = 16,
):
    """S1 标准训练主函数 (自动断点续训).

    Args:
        dataloader_kwargs: 传递给 create_dataloaders 的额外参数.
        val_batch_mult: 验证/测试 loader 的 batch 倍数. 只影响前向分批,
            全部指标在拼接后的完整预测上计算, 与该值无关 (>1 仅为提速).
        batch_size: 训练 batch. 默认 16; PointNeXt S1 在 16GB 卡上 b16 会 OOM
            (ball_query 排序需 >16GB), 该骨干单独以 b8 补跑, 稿件中需注明协议偏差.
    """
    from lib.utils import CHECKPOINT_BASE

    set_seed(42)
    dataloader_kwargs = dataloader_kwargs or {}

    console.rule(f"[bold]{desc}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}  Arch: {arch}  Epochs: {num_epochs}  Batch: {batch_size}")

    if ckpt_dir is None:
        ckpt_dir = os.path.join(CHECKPOINT_BASE, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 数据 ──
    make_train, make_val, make_test = create_dataloaders(
        8192,
        batch_size,
        keep_ratio=0.4,
        samples_per_epoch=3000,
        val_batch_mult=val_batch_mult,
        **dataloader_kwargs,
    )

    # ── 模型 ──
    model = make_s1_model(arch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  [dim]参数: {n_params:,}[/]")

    # ── 优化器 + 调度器 ──
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=num_epochs)

    # ── 断点续训 (自动检测) ──
    ckpt_path = os.path.join(ckpt_dir, "best_s1.pth")
    start_ep = 1
    best_m = 0.0
    best_ep = 0
    patience_counter = 0
    hist = build_history_template_s1()

    if os.path.exists(ckpt_path):
        saved_ep = load_checkpoint(ckpt_path, model, opt, sch)
        start_ep = saved_ep + 1
        hist_path = os.path.join(ckpt_dir, f"history_{tag}.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist = json.load(f)
            # 按与训练循环一致的 mIoU 口径恢复 best_m
            # (history 未存 mIoU, 从逐类 IoU 重建; cm 与 mIoU 口径不同, 不能混用)
            if hist.get("epoch"):
                mious = [
                    sum(hist[f"iou_{n}"][i] for n in S1_NAMES.values()) / len(S1_NAMES)
                    for i in range(len(hist["epoch"]))
                ]
                best_m = max(mious)
                best_ep = int(hist["epoch"][mious.index(best_m)])
        console.print(
            f"  [green]断点续训: 从 epoch {start_ep} 继续 (最佳 cm={best_m:.3f} @ ep{best_ep})[/]"
        )

    # ── 训练循环 ──
    for ep in range(start_ep, num_epochs + 1):
        if ep > 1:
            sch.step()

        tl = train_one_epoch(
            model,
            make_train(),
            criterion,
            opt,
            ep,
            tag=tag,
            forward_and_labels_fn=_s1_forward_and_labels,
        )
        vl, ious, acc, prf, cm = validate_s1(model, make_val, criterion)
        m = ious["mIoU"]

        fill_history_s1(hist, ep, tl, vl, ious, acc, prf, cm)

        io = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
        console.print(
            f"  {tag}: tl={tl:.4f} vl={vl:.4f} acc={acc:.3f} mIoU={m:.3f} | [{io}] | cm={cm:.3f}"
        )

        if m > best_m:
            best_m = m
            best_ep = ep
            patience_counter = 0
            save_checkpoint(ckpt_path, model, opt, sch, epoch=ep, best_miou=best_m)
            console.print(f"  [green]↑ 最佳 (mIoU={m:.3f})[/]")
        else:
            patience_counter += 1
            console.print(
                f"  [dim]早停计数: {patience_counter}/{early_stop_patience} (最佳 @ ep{best_ep}: mIoU={best_m:.3f})[/]"
            )

        save_history(ckpt_dir, tag, hist)

        if patience_counter >= early_stop_patience:
            console.print(
                f"[yellow]早停触发: {early_stop_patience} epoch 无提升 (最佳 @ ep{best_ep})[/]"
            )
            break

    # ── 测试集 ──
    console.rule("[bold blue]测试集[/]")
    load_checkpoint(ckpt_path, model)
    tl, ious, acc, prf, cm = validate_s1(model, make_test, criterion)
    tio = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
    console.print(
        f"test_loss={tl:.4f} acc={acc:.3f} mIoU={ious['mIoU']:.3f} | [{tio}] | cm={cm:.3f}"
    )

    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "arch": arch,
                "best_miou": best_m,
                "test_miou": ious["mIoU"],
                "test_cm": cm,
                **serialize_metrics(ious, prf),
            },
            f,
            indent=2,
        )
    console.print(
        f"[bold green]完成! 最佳 mIoU={best_m:.3f}, 测试 mIoU={ious['mIoU']:.3f}[/]"
    )


def train_s1_radius(
    arch: str = "pointmlp",
    radius: float = 150.0,
    num_epochs: int = 120,
    tag: str = "S1_radius",
    desc: str = "S1 半径敏感性",
    ckpt_dir: str = None,
):
    """S1 半径敏感性训练."""
    import lib.utils.datasets as ds_module
    from lib.utils import CHECKPOINT_BASE

    set_seed(42)
    if ckpt_dir is None:
        ckpt_dir = os.path.join(
            CHECKPOINT_BASE, "sensitivity_S1_radius", f"{tag}_{arch}"
        )
    os.makedirs(ckpt_dir, exist_ok=True)
    console.rule(f"[bold]{desc}: 半径={radius:.0f}mm[/]")

    ds_module.PATCH_RADIUS_MM = radius
    make_train, make_val, make_test = create_dataloaders(
        8192,
        16,
        keep_ratio=0.4,
        samples_per_epoch=3000,
    )

    model = make_s1_model(arch).to(DEVICE)
    console.print(
        f"  [dim]参数: {sum(p.numel() for p in model.parameters()):,} | 半径: {radius:.0f}mm[/]"
    )

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=num_epochs)

    best_m = 0.0
    best_ep = 0
    patience_counter = 0
    hist = build_history_template_s1()

    for ep in range(1, num_epochs + 1):
        if ep > 1:
            sch.step()

        tl = train_one_epoch(
            model,
            make_train(),
            criterion,
            opt,
            ep,
            tag=tag,
            forward_and_labels_fn=_s1_forward_and_labels,
        )
        vl, ious, acc, prf, cm = validate_s1(model, make_val, criterion)
        m = ious["mIoU"]

        fill_history_s1(hist, ep, tl, vl, ious, acc, prf, cm)

        io = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
        console.print(
            f"  {tag}: tl={tl:.4f} vl={vl:.4f} acc={acc:.3f} mIoU={m:.3f} | [{io}] | cm={cm:.3f}"
        )

        if m > best_m:
            best_m = m
            best_ep = ep
            patience_counter = 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_s1.pth"), model, opt, sch, epoch=ep
            )
            console.print(f"  [green]↑ 最佳 (mIoU={m:.3f})[/]")
        else:
            patience_counter += 1

        save_history(ckpt_dir, tag, hist)
        if patience_counter >= 10:
            console.print(f"[yellow]早停触发 @ ep{best_ep}[/]")
            break

    load_checkpoint(os.path.join(ckpt_dir, "best_s1.pth"), model)
    tl, ious, acc, prf, cm = validate_s1(model, make_test, criterion)
    io = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
    console.print(
        f"  test: loss={tl:.4f} acc={acc:.3f} mIoU={ious['mIoU']:.3f} | [{io}] | cm={cm:.3f}"
    )

    summary = {
        "radius_mm": radius,
        "best_val_miou": best_m,
        "test_miou": ious["mIoU"],
        "test_cm": cm,
        **serialize_metrics(ious, prf),
    }
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ds_module.PATCH_RADIUS_MM = 150.0
    return summary
