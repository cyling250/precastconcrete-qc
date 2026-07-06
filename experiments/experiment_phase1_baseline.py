"""
Phase 1 Baseline: S1 三分类 (噪声/混凝土/金属).

使用方法:
  python experiment_phase1_baseline.py                              # PointNet++ (默认)
  python experiment_phase1_baseline.py --arch pointnet               # PointNet
  python experiment_phase1_baseline.py --arch pointmlp               # PointMLP
  python experiment_phase1_baseline.py --arch pointnext              # PointNeXt

设计哲学:
  - 固定半径 (scale=None → 使用默认 0.2/0.4)
  - 仅使用 XYZ 坐标 (in_channels=3, 忽略 geo_features)
  - CrossEntropyLoss
  - AdamW + CosineAnnealingLR

标签映射: 原始 4 类 → S1 3 类
  钢筋(2)→金属(2), 套筒(3)→金属(2)
"""

import os, json, argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from _common import (
    console,
    NUM_POINTS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    DEVICE,
    CHECKPOINT_BASE,
    set_seed,
    _serialize_metrics,
    _save_history,
    create_dataloaders,
)
from lib.pointnet2.models import (
    PointNetPlusPlus, PointNetSeg, PointMLPSeg, PointNeXtSeg,
    count_params,
)
from lib.pointnet2.metrics_utils import (
    compute_iou,
    compute_precision_recall_f1,
    compute_accuracy,
)

# ── 架构注册表 ──
ARCH_REGISTRY = {
    "pointnet++": PointNetPlusPlus,
    "pointnet": PointNetSeg,
    "pointmlp": PointMLPSeg,
    "pointnext": PointNeXtSeg,
}

EPOCHS = 80
EARLY_STOP_PATIENCE = 10      # cm 连续 10 epoch 不创新高则停止
LR = 1e-3
S1_NAMES = {0: "噪声", 1: "混凝土", 2: "金属"}


# ══════════════════════════════════════════════════════
#  S1 模型 (固定半径, 仅 XYZ)
# ══════════════════════════════════════════════════════

class S1Model(nn.Module):
    """S1 三分类统一包装器 — 固定半径, 仅 XYZ, 可替换骨干."""

    def __init__(self, arch: str):
        super().__init__()
        self.arch = arch
        model_cls = ARCH_REGISTRY[arch]
        # S1 仅使用 XYZ(3), 不传入 geo_features
        self.backbone = model_cls(num_classes=3, in_channels=3)

    def forward(self, xyz, geo_features=None, scale=None):
        # scale=None → 固定半径 (对 PointNet 等无半径架构无影响)
        # geo_features=None → 仅用 XYZ
        return self.backbone(xyz, geo_features=None, scale=None)


def make_s1_model(arch: str) -> S1Model:
    """创建 S1 模型."""
    return S1Model(arch)


# ══════════════════════════════════════════════════════
#  标签映射
# ══════════════════════════════════════════════════════

def _map_labels(lbl: torch.Tensor) -> torch.Tensor:
    """4 类标签 → 3 类: 钢筋(2)→金属(2), 套筒(3)→金属(2)."""
    clbl = lbl.clone()
    clbl[clbl == 3] = 2
    return clbl


# ══════════════════════════════════════════════════════
#  验证
# ══════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader_fn, criterion):
    model.eval()
    loader = loader_fn()
    total_loss = 0.0
    all_p, all_t = [], []
    prog = Progress(
        TextColumn("[bold]val[/]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TimeRemainingColumn(),
        console=console,
    )
    ttask = prog.add_task("val", total=len(loader), loss=0)
    with torch.no_grad(), prog:
        for pts, feat, lbl, scale in loader:
            pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
            logits = model(pts, scale=None)
            clbl = _map_labels(lbl)
            loss = criterion(logits.permute(0, 2, 1), clbl)
            total_loss += loss.item()
            all_p.append(logits.cpu())
            all_t.append(clbl.cpu())
            prog.update(ttask, advance=1, loss=loss.item())
    p, t = torch.cat(all_p), torch.cat(all_t)
    ious = compute_iou(p, t, S1_NAMES)
    acc = compute_accuracy(p, t)
    prf = compute_precision_recall_f1(p, t, S1_NAMES)
    cm = (ious["混凝土"] + ious["金属"]) / 2.0
    return total_loss / len(loader), ious, acc, prf, cm


# ══════════════════════════════════════════════════════
#  CL
# ══════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: S1 三分类基线")
    parser.add_argument("--arch", type=str, default="pointnet++",
                        choices=list(ARCH_REGISTRY.keys()),
                        help="骨干网络架构 (默认: pointnet++)")
    parser.add_argument("--run-suffix", type=str, default="",
                        help="自定义标签后缀 (用于区分不同运行, 不覆盖旧结果)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖训练轮数 (默认使用 EPOCHS=80)")
    parser.add_argument("--patience", type=int, default=None,
                        help="早停耐心值 (默认使用 EARLY_STOP_PATIENCE=10)")
    parser.add_argument("--resume", action="store_true",
                        help="从 checkpoints 目录加载 best_s1.pth 断点续训")
    return parser.parse_args()


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════

def main():
    args = parse_args()
    arch = args.arch
    run_suffix = args.run_suffix
    tag_suffix = f"_{arch}" if arch != "pointnet++" else ""
    tag_suffix += run_suffix
    tag = f"phase1_baseline{tag_suffix}"
    desc = f"Phase 1: S1 ({arch}, 噪声/混凝土/金属)"

    num_epochs = args.epochs if args.epochs is not None else EPOCHS
    early_stop_patience = args.patience if args.patience is not None else EARLY_STOP_PATIENCE

    CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, tag)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    set_seed(42)
    console.rule(f"[bold]{desc}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}  Arch: {arch}  Epochs: {num_epochs}  Patience: {early_stop_patience}")
    if run_suffix:
        console.print(f"[yellow]  运行标识: {run_suffix}, 保存至: {tag}[/]")

    make_train, make_val, make_test = create_dataloaders(
        NUM_POINTS,
        BATCH_SIZE,
        keep_ratio=0.4,
        samples_per_epoch=3000,
    )

    model = make_s1_model(arch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  [dim]参数: {n_params:,}[/]")

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=num_epochs)

    best_m = 0.0
    best_ep = 0
    patience_counter = 0
    hist = {"epoch": [], "loss": [], "val_loss": [], "acc": [], "cm": []}
    start_epoch = 1

    # ── 断点续训 ──
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_s1.pth")
    hist_path = os.path.join(CHECKPOINT_DIR, f"history_{tag}.json")
    if args.resume:
        if not os.path.exists(ckpt_path):
            console.print(f"[red]未找到 checkpoint: {ckpt_path}, 从头开始训练[/]")
        else:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            console.print(f"  [green]已加载权重: {ckpt_path}[/]")
            if os.path.exists(hist_path):
                import json as _json
                with open(hist_path) as f:
                    old_hist = _json.load(f)
                completed = max(old_hist.get("epoch", [0]))
                start_epoch = completed + 1
                # 恢复历史记录和最佳跟踪
                hist = old_hist
                cm_vals = hist.get("cm", [0.0])
                if cm_vals:
                    best_idx = int(max(range(len(cm_vals)), key=lambda i: cm_vals[i]))
                    best_ep = hist["epoch"][best_idx]
                # 设置 scheduler 到正确位置
                for _ in range(start_epoch - 1):
                    sch.step()
                console.print(f"  [green]断点续训: 从 epoch {start_epoch} 继续 (已完成 {completed} epoch)[/]")

    for ep in range(start_epoch, num_epochs + 1):
        if ep > 1:
            sch.step()
        model.train()
        train_loss = 0.0
        vb = 0
        prog = Progress(
            TextColumn(f"[bold]S1-{arch}-ep{ep:03d}[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TimeRemainingColumn(),
            console=console,
        )
        ttask = prog.add_task("s1", total=len(make_train()), loss=0)
        with prog:
            for pts, feat, lbl, scale in make_train():
                pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
                opt.zero_grad()
                logits = model(pts, scale=None)
                clbl = _map_labels(lbl)
                loss = criterion(logits.permute(0, 2, 1), clbl)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                train_loss += loss.item()
                vb += 1
                prog.update(ttask, advance=1, loss=loss.item())

        tl = train_loss / max(vb, 1)
        vl, ious, acc, prf, cm = validate(model, make_val, criterion)
        m = ious["mIoU"]
        hist["epoch"].append(ep)
        hist["loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["acc"].append(acc)
        hist["cm"].append(cm)
        io = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
        pr = " | ".join(
            f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
            for k, v in prf.items()
        )
        console.print(
            f"  S1: tl={tl:.4f} vl={vl:.4f} acc={acc:.3f} mIoU={m:.3f} | [{io}] | cm={cm:.3f}"
        )
        console.print(f"    PRF: [{pr}]")
        if m > best_m:
            best_m = m
            best_ep = ep
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_s1.pth"))
            console.print(f"  [green]↑ 最佳 (mIoU={m:.3f})[/]")
        else:
            patience_counter += 1
            console.print(f"  [dim]早停计数: {patience_counter}/{early_stop_patience} (最佳 @ ep{best_ep}: mIoU={best_m:.3f})[/]")
        _save_history(CHECKPOINT_DIR, tag, hist)

        if patience_counter >= early_stop_patience:
            console.print(f"[yellow]早停触发: {early_stop_patience} epoch 无提升, 停止训练 (最佳 @ ep{best_ep})[/]")
            break

    # ── 测试集 ──
    console.rule("[bold blue]测试集[/]")
    model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "best_s1.pth"), map_location=DEVICE)
    )
    tl, ious, acc, prf, cm = validate(model, make_test, criterion)
    tio = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
    console.print(
        f"test_loss={tl:.4f} acc={acc:.3f} mIoU={ious['mIoU']:.3f} | [{tio}] | cm={cm:.3f}"
    )
    pr_str = " | ".join(
        f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
        for k, v in prf.items()
    )
    console.print(f"  PRF: [{pr_str}]")

    with open(os.path.join(CHECKPOINT_DIR, "summary_phase1.json"), "w") as f:
        json.dump(
            {
                "arch": arch,
                "best_miou": best_m,
                "test_miou": ious["mIoU"],
                "test_cm": cm,
                **_serialize_metrics(ious, prf),
            },
            f,
            indent=2,
        )
    console.print(f"[bold green]完成! 最佳 mIoU={best_m:.3f}[/]")


if __name__ == "__main__":
    main()
