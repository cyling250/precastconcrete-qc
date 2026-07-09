"""
S2 训练器: 级联管线精分割 (噪声/混凝土/钢筋/套筒).

管线: S1 冻结 → 预计算 S2 缓存 → S2 训练.
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
    CLASS_NAMES,
    set_seed,
    serialize_metrics,
    create_dataloaders,
    train_one_epoch,
    build_history_template_s2,
    fill_history_s2,
    save_history,
    save_checkpoint,
    load_checkpoint,
)
from lib.cascade import (
    Stage2Model,
    build_s2_dataset,
    load_s2_loader,
    val_v2_simple,
    val_s2_cached,
    print_v2,
)
from experiments._s1_trainer import make_s1_model

VOXEL_MM = 40.0
VAL_INTERVAL = 2


def _s2_forward_and_labels(model, batch):
    """S2 6ch 前向: ignore_index=1 (不训练混凝土)."""
    pts_pad, s1p_pad, lbl_pad, scale_bt = batch
    pts_pad = pts_pad.to(DEVICE)
    s1p_pad = s1p_pad.to(DEVICE)
    lbl_pad = lbl_pad.to(DEVICE)
    logits = model(pts_pad, s1p_pad, scale=scale_bt)
    loss_lbl = lbl_pad.clone()
    loss_lbl[loss_lbl == 1] = -1  # ignore concrete
    return logits, loss_lbl


def _s1_ckpt_dir(arch: str) -> str:
    from lib.utils import CHECKPOINT_BASE

    return os.path.join(CHECKPOINT_BASE, f"phase1_baseline_{arch}")


def _s2_ckpt_dir(arch: str) -> str:
    from lib.utils import CHECKPOINT_BASE

    return os.path.join(CHECKPOINT_BASE, f"phase2_baseline_{arch}")


def _load_s1(arch: str, s1_ckpt_path: str):
    """加载冻结的 S1 模型."""
    if not os.path.exists(s1_ckpt_path):
        console.print(f"[red]未找到 S1 权重 {s1_ckpt_path}[/]")
        return None
    s1_model = make_s1_model(arch).to(DEVICE)
    state = torch.load(s1_ckpt_path, map_location=DEVICE)
    sd = state.get("model_state", state)  # 兼容新旧格式
    s1_model.load_state_dict(sd)
    s1_model.eval()
    for p in s1_model.parameters():
        p.requires_grad = False
    return s1_model


def train_s2(
    arch: str = "pointnet++",
    num_epochs: int = 80,
    voxel_mm: float = VOXEL_MM,
    val_interval: int = VAL_INTERVAL,
    tag: str = None,
    desc: str = None,
    ckpt_dir: str = None,
):
    """S2 标准训练: S1 冻结 → 预计算缓存 → S2 训练 (自动断点续训)."""
    from lib.utils import CHECKPOINT_BASE

    if tag is None:
        tag = f"phase2_baseline_{arch}"
    if desc is None:
        desc = f"Phase 2: S2 ({arch}, 噪声/混凝土/钢筋/套筒, S1前馈)"
    if ckpt_dir is None:
        ckpt_dir = _s2_ckpt_dir(arch)
    os.makedirs(ckpt_dir, exist_ok=True)

    s1_ckpt_path = os.path.join(_s1_ckpt_dir(arch), "best_s1.pth")
    train_cache = os.path.join(ckpt_dir, "s2_train.pt")
    val_cache = os.path.join(ckpt_dir, "s2_val.pt")
    test_cache = os.path.join(ckpt_dir, "s2_test.pt")

    set_seed(42)
    console.rule(f"[bold]{desc}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}  S1 Arch: {arch}")
    console.print(f"  体素: {voxel_mm}mm, 仅混凝土降采样, 保留全部噪声+金属")

    # ── 加载 S1 ──
    s1_model = _load_s1(arch, s1_ckpt_path)
    if s1_model is None:
        return
    console.print(f"  [dim]S1 ({arch}) 已加载 (冻结)[/]")

    # ── 数据加载器 ──
    make_train, make_val, make_test = create_dataloaders(
        8192,
        16,
        keep_ratio=0.4,
        samples_per_epoch=3000,
    )

    # ── 预计算 S2 数据集 (训练前一次性完成) ──
    console.rule("[bold]预计算 S2 数据集[/]")
    for loader_fn, cache_path in [
        (make_train, train_cache),
        (make_val, val_cache),
        (make_test, test_cache),
    ]:
        build_s2_dataset(
            s1_model, loader_fn, cache_path, DEVICE, console, voxel_mm=voxel_mm
        )

    train_loader = load_s2_loader(train_cache, 16, shuffle=True)
    val_loader = load_s2_loader(val_cache, 16, shuffle=False)
    n_train = len(torch.load(train_cache, weights_only=False))
    console.print(
        f"  [bold]训练:[/] {n_train} patches ({len(train_loader)} batch/epoch)"
    )

    # ── S2 模型 ──
    s2_model = Stage2Model().to(DEVICE)
    n_p = sum(p.numel() for p in s2_model.parameters())
    console.print(f"  [dim]S2 参数: {n_p:,}[/]")

    # ── 优化器 + 调度器 ──
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.AdamW(s2_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=num_epochs)

    # ── 断点续训 (自动检测) ──
    ckpt_path = os.path.join(ckpt_dir, "best_s2.pth")
    start_ep = 1
    best_cm = 0.0
    hist = build_history_template_s2()

    if os.path.exists(ckpt_path):
        saved_ep = load_checkpoint(ckpt_path, s2_model, opt, sch)
        start_ep = saved_ep + 1
        hist_path = os.path.join(ckpt_dir, f"history_{tag}.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                hist = json.load(f)
            cm_vals = hist.get("val_cm", [])
            if cm_vals:
                best_cm = max(cm_vals)
        console.print(
            f"  [green]断点续训: 从 epoch {start_ep} 继续 (最佳 cm={best_cm:.3f})[/]"
        )

    # 缓存上一次验证结果 (用于非验证 interval 的 epoch)
    hist_c_ious = hist_c_prf = hist_c_acc = hist_s_ious = hist_rs_iou = hist_cm = None

    # ── 训练循环 ──
    for ep in range(start_ep, num_epochs + 1):
        if ep > 1:
            sch.step()

        s2_tl = train_one_epoch(
            s2_model,
            train_loader,
            criterion,
            opt,
            ep,
            tag=f"S2-{arch}",
            forward_and_labels_fn=_s2_forward_and_labels,
        )

        do_val = (ep == 1) or (ep % val_interval == 0)
        if do_val:
            c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_s2_cached(
                s2_model,
                val_loader,
                DEVICE,
                CLASS_NAMES,
            )
            print_v2(
                ep, s2_tl, c_ious, c_prf, c_acc, s_ious, rs_iou, cm, console=console
            )
        else:
            c_ious = hist_c_ious
            c_prf = hist_c_prf
            c_acc = hist_c_acc
            s_ious = hist_s_ious
            rs_iou = hist_rs_iou
            cm = hist_cm

        if c_ious is not None:
            fill_history_s2(hist, ep, s2_tl, c_ious, c_prf, c_acc, s_ious, rs_iou, cm)

        if do_val and cm > best_cm:
            best_cm = cm
            save_checkpoint(
                ckpt_path,
                s2_model,
                opt,
                sch,
                epoch=ep,
                cm=cm,
                c_ious=c_ious,
                rs_iou=rs_iou,
            )
            console.print(f"  [green]↑ 最佳 (cm={cm:.3f})[/]")

        if do_val:
            hist_c_ious = c_ious
            hist_c_prf = c_prf
            hist_c_acc = c_acc
            hist_s_ious = s_ious
            hist_rs_iou = rs_iou
            hist_cm = cm

        save_history(ckpt_dir, tag, hist)

    # ── 测试集 ──
    console.rule("[bold blue]测试集 (S1→S2 全管线)[/]")
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, s2_model)
        c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
            s1_model,
            s2_model,
            make_test,
            DEVICE,
            CLASS_NAMES,
            voxel_mm=voxel_mm,
        )
        cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
        console.print(
            f"  test_acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}"
        )

    summary = {
        "arch": arch,
        "best_cm": best_cm,
        "test_miou": c_ious.get("mIoU", 0) if os.path.exists(ckpt_path) else 0,
        "test_cm": cm if os.path.exists(ckpt_path) else 0,
        "test_rs_iou": rs_iou if os.path.exists(ckpt_path) else 0,
    }
    if os.path.exists(ckpt_path):
        summary.update(serialize_metrics(c_ious, c_prf))
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[bold green]完成! 最佳 cm={best_cm:.3f}[/]")


def train_s2_voxel(
    arch: str = "pointmlp",
    voxel_mm: float = 40.0,
    num_epochs: int = 60,
    tag: str = "voxel",
    desc: str = "S2 体素敏感性",
):
    """S2 体素大小敏感性训练 (简化版)."""
    from lib.utils import CHECKPOINT_BASE

    set_seed(42)
    ckpt_dir = os.path.join(CHECKPOINT_BASE, f"sensitivity_{tag}_pointmlp")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_cache = os.path.join(ckpt_dir, "s2_train.pt")
    val_cache = os.path.join(ckpt_dir, "s2_val.pt")

    console.rule(f"[bold]{desc}[/]")
    s1_ckpt = os.path.join(CHECKPOINT_BASE, "phase1_baseline_pointmlp", "best_s1.pth")
    if not os.path.exists(s1_ckpt):
        console.print(f"[red]未找到 S1 权重 {s1_ckpt}[/]")
        return None

    s1_model = make_s1_model(arch).to(DEVICE)
    state = torch.load(s1_ckpt, map_location=DEVICE)
    if "model_state" in state:
        s1_model.load_state_dict(state["model_state"])
    else:
        s1_model.load_state_dict(state)
    s1_model.eval()
    for p in s1_model.parameters():
        p.requires_grad = False

    make_train, make_val, make_test = create_dataloaders(
        8192,
        16,
        keep_ratio=0.4,
        samples_per_epoch=3000,
    )

    build_s2_dataset(
        s1_model, make_train, train_cache, DEVICE, console, voxel_mm=voxel_mm
    )
    build_s2_dataset(s1_model, make_val, val_cache, DEVICE, console, voxel_mm=voxel_mm)
    train_loader = load_s2_loader(train_cache, 16, shuffle=True)
    val_loader = load_s2_loader(val_cache, 32, shuffle=False)

    s2_model = Stage2Model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.AdamW(s2_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=num_epochs)

    best_cm = 0.0
    for ep in range(1, num_epochs + 1):
        if ep > 1:
            sch.step()
        tl = train_one_epoch(
            s2_model,
            train_loader,
            criterion,
            opt,
            ep,
            tag=tag,
            forward_and_labels_fn=_s2_forward_and_labels,
        )
        c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
            s1_model,
            s2_model,
            make_val,
            DEVICE,
            CLASS_NAMES,
            voxel_mm=voxel_mm,
        )
        console.print(
            f"  {tag} ep{ep:03d}: tl={tl:.4f} acc={c_acc:.3f} mIoU={c_ious.get('mIoU', 0):.3f} cm={cm:.3f} rsIoU={rs_iou:.3f}"
        )
        if cm > best_cm:
            best_cm = cm
            save_checkpoint(
                os.path.join(ckpt_dir, "best_s2.pth"),
                s2_model,
                opt,
                sch,
                epoch=ep,
                cm=cm,
            )

    load_checkpoint(os.path.join(ckpt_dir, "best_s2.pth"), s2_model)
    c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
        s1_model,
        s2_model,
        make_test,
        DEVICE,
        CLASS_NAMES,
        voxel_mm=voxel_mm,
    )
    console.print(
        f"  [bold]test: acc={c_acc:.3f} mIoU={c_ious.get('mIoU', 0):.3f} cm={cm:.3f} rsIoU={rs_iou:.3f}[/]"
    )
    return {
        "voxel_mm": voxel_mm,
        "test_miou": c_ious.get("mIoU", 0),
        "test_cm": cm,
        "test_rs_iou": rs_iou,
    }
