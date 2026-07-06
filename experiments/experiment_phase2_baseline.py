"""
Phase 2 Baseline: S2 四类精分割 (噪声/混凝土/钢筋/套筒).

管线: S1 → 混凝土4cm体素降采样 → S2

使用方法:
  python experiment_phase2_baseline.py                              # PointNet++ (默认)
  python experiment_phase2_baseline.py --arch pointnet               # S1=PointNet
  python experiment_phase2_baseline.py --arch pointmlp               # S1=PointMLP
  python experiment_phase2_baseline.py --arch pointnext              # S1=PointNeXt

S2 输入: XYZ(3) + S1概率(3) = 6通道 (固定为 PointNet++)
S2 训练: 不训练混凝土 (ignore_index=1), 只修正噪声/钢筋/套筒
S2 推理: 固定混凝土为输入标签, S2 仅输出噪声/钢筋/套筒

训练: 全程使用 S1 预测, 预计算 + 缓存
验证: S1 管线 → S2 管线, 组合指标评估
"""

import os, json, time, argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from _common import (
    console, NUM_POINTS, BATCH_SIZE, LR, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_BASE, set_seed, _serialize_metrics, _save_history,
    create_dataloaders, CLASS_NAMES,
)
from experiment_phase1_baseline import make_s1_model
from lib.pointnet2.doublephase import (
    Stage2Model, build_s2_dataset, load_s2_loader,
    val_v2_simple, print_v2,
)

EPOCHS = 80
VOXEL_MM = 40.0       # 4cm 体素 (mm)

ARCH_REGISTRY = ["pointnet++", "pointnet", "pointmlp", "pointnext"]


def _s1_ckpt_dir(arch: str) -> str:
    """返回对应架构的 S1 checkpoint 目录."""
    suffix = "" if arch == "pointnet++" else f"_{arch}"
    return os.path.join(CHECKPOINT_BASE, f"phase1_baseline{suffix}")


def _s2_ckpt_dir(arch: str) -> str:
    """返回对应架构的 S2 checkpoint 目录."""
    suffix = "" if arch == "pointnet++" else f"_{arch}"
    return os.path.join(CHECKPOINT_BASE, f"phase2_baseline{suffix}")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: S2 四类精分割基线")
    parser.add_argument("--arch", type=str, default="pointnet++",
                        choices=ARCH_REGISTRY,
                        help="S1 骨干网络架构 (默认: pointnet++)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    arch = args.arch

    tag_suffix = "" if arch == "pointnet++" else f"_{arch}"
    tag = f"phase2_baseline{tag_suffix}"
    desc = f"Phase 2: S2 ({arch}, 噪声/混凝土/钢筋/套筒, S1前馈)"
    ckpt_dir = _s2_ckpt_dir(arch)
    os.makedirs(ckpt_dir, exist_ok=True)

    s1_ckpt_path = os.path.join(_s1_ckpt_dir(arch), "best_s1.pth")
    train_cache = os.path.join(ckpt_dir, "s2_train.pt")
    val_cache = os.path.join(ckpt_dir, "s2_val.pt")
    test_cache = os.path.join(ckpt_dir, "s2_test.pt")

    set_seed(42)
    console.rule(f"[bold]{desc}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}  S1 Arch: {arch}")
    console.print(f"  体素: {VOXEL_MM}mm (4cm), 仅混凝土降采样, 保留全部噪声+金属")

    # ── 检查 S1 权重 ──
    if not os.path.exists(s1_ckpt_path):
        console.print(f"[red]未找到 S1 权重 {s1_ckpt_path}, 请先运行 experiment_phase1_baseline.py --arch {arch}[/]")
        return

    # ── S1 模型 (冻结, 仅推理) ──
    s1_model = make_s1_model(arch).to(DEVICE)
    s1_model.load_state_dict(torch.load(s1_ckpt_path, map_location=DEVICE))
    s1_model.eval()
    for p in s1_model.parameters():
        p.requires_grad = False
    console.print(f"  [dim]S1 ({arch}) 已加载 (冻结)[/]")

    # ── 数据加载 ──
    make_train, make_val, make_test = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4, samples_per_epoch=3000,
    )

    # ── 预计算 S2 数据集 (全管线: S1 预测指导降采样) ──
    console.rule("[bold]预计算 S2 数据集[/]")
    build_s2_dataset(s1_model, make_train, train_cache, DEVICE, console,
                     voxel_mm=VOXEL_MM)
    build_s2_dataset(s1_model, make_val, val_cache, DEVICE, console,
                     voxel_mm=VOXEL_MM)
    build_s2_dataset(s1_model, make_test, test_cache, DEVICE, console,
                     voxel_mm=VOXEL_MM)

    train_loader = load_s2_loader(train_cache, BATCH_SIZE, shuffle=True)

    n_train = len(torch.load(train_cache, weights_only=False))
    console.print(f"  [bold]训练:[/] {n_train} patches ({len(train_loader)} batch/epoch)")

    # ── S2 模型 ──
    s2_model = Stage2Model().to(DEVICE)
    n_p = sum(p.numel() for p in s2_model.parameters())
    console.print(f"  [dim]S2 参数: {n_p:,}[/]")

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.AdamW(s2_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS)

    best_cm = 0.0
    hist = {
        "epoch": [], "loss": [], "val_loss": [],
        "val_acc": [], "val_cm": [], "val_rs_iou": [],
    }
    # 每类 IoU + PRF
    for cls_name in CLASS_NAMES.values():
        hist[f"iou_{cls_name}"] = []
        for met in ["precision", "recall", "f1"]:
            hist[f"prf_{cls_name}_{met}"] = []
    # S2 纯指标 (噪声/钢筋/套筒)
    for tag in ["S2_噪声", "S2_钢筋", "S2_套筒"]:
        hist[f"s2pure_{tag}"] = []

    for ep in range(1, EPOCHS + 1):
        if ep > 1:
            sch.step()
        s2_model.train()
        tl = 0.0
        n_b = 0
        prog = Progress(
            TextColumn(f"[bold]S2-{arch}-ep{ep:03d}[/]"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TimeRemainingColumn(), console=console,
        )
        tt = prog.add_task("s2", total=len(train_loader), loss=0.0)
        with prog:
            for pts_pad, s1p_pad, lbl_pad, scale_bt in train_loader:
                pts_pad = pts_pad.to(DEVICE)
                s1p_pad = s1p_pad.to(DEVICE)
                lbl_pad = lbl_pad.to(DEVICE)
                scale_bt = scale_bt.to(DEVICE)

                opt.zero_grad()
                logits = s2_model(pts_pad, s1p_pad, scale=scale_bt)

                loss_lbl = lbl_pad.clone()
                loss_lbl[loss_lbl == 1] = -1
                loss = criterion(logits.permute(0, 2, 1), loss_lbl)

                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(s2_model.parameters(), max_norm=1.0)
                opt.step()
                tl += loss.item()
                n_b += 1
                prog.update(tt, advance=1, loss=loss.item())

        s2_tl = tl / max(n_b, 1)

        # ── 验证 (全管线) ──
        c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
            s1_model, s2_model, make_val, DEVICE, CLASS_NAMES, voxel_mm=VOXEL_MM,
        )
        print_v2(ep, s2_tl, c_ious, c_prf, c_acc, s_ious, rs_iou, cm, console=console)

        hist["epoch"].append(ep)
        hist["loss"].append(s2_tl)
        hist["val_loss"].append(c_ious.get("mIoU", 0))
        hist["val_acc"].append(c_acc)
        hist["val_cm"].append(cm)
        hist["val_rs_iou"].append(rs_iou)
        # 保存每类 IoU
        for cls_name in CLASS_NAMES.values():
            hist[f"iou_{cls_name}"].append(c_ious.get(cls_name, 0.0))
        # 保存每类 PRF
        for cls_name in CLASS_NAMES.values():
            pm = c_prf.get(cls_name, {})
            hist[f"prf_{cls_name}_precision"].append(pm.get("precision", 0.0))
            hist[f"prf_{cls_name}_recall"].append(pm.get("recall", 0.0))
            hist[f"prf_{cls_name}_f1"].append(pm.get("f1", 0.0))
        # S2 纯指标
        for tag in ["S2_噪声", "S2_钢筋", "S2_套筒"]:
            hist[f"s2pure_{tag}"].append(s_ious.get(tag, 0.0))

        if cm > best_cm:
            best_cm = cm
            torch.save({
                "epoch": ep,
                "model_state": s2_model.state_dict(),
                "opt_state": opt.state_dict(),
                "cm": cm,
                "c_ious": c_ious,
                "rs_iou": rs_iou,
            }, os.path.join(ckpt_dir, "best_s2.pth"))
            console.print(f"  [green]↑ 最佳 (cm={cm:.3f})[/]")

        _save_history(ckpt_dir, tag, hist)

    # ── 测试集 ──
    console.rule("[bold blue]测试集 (S1→S2 全管线)[/]")
    ckpt = os.path.join(ckpt_dir, "best_s2.pth")
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=DEVICE)
        s2_model.load_state_dict(sd["model_state"])
        c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
            s1_model, s2_model, make_test, DEVICE, CLASS_NAMES, voxel_mm=VOXEL_MM,
        )
        cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
        console.print(f"  test_acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}")
    else:
        console.print("[yellow]未找到 best_s2.pth[/]")

    with open(os.path.join(ckpt_dir, "summary_phase2.json"), "w") as f:
        json.dump({
            "arch": arch,
            "best_cm": best_cm,
            "test_miou": c_ious.get("mIoU", 0),
            "test_cm": cm,
            "test_rs_iou": rs_iou,
        }, f, indent=2)
    console.print(f"[bold green]完成! 最佳 cm={best_cm:.3f}[/]")


if __name__ == "__main__":
    main()
