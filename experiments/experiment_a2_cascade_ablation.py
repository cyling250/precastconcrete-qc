"""
消融 A2 — 无级联管线: 单阶段 4 类 PointNet++ 直接分割.

验证级联自适应体素降采样管线的必要性。
使用 PointNet++ 固定半径 + CrossEntropyLoss + 4 类分割。
本质是单阶段方法，和 baseline cascade 对比。
"""

import os, json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from _common import (
    console, NUM_POINTS, BATCH_SIZE, LR, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_BASE, NUM_CLASSES, CLASS_NAMES, set_seed,
    _serialize_metrics, _save_history, create_dataloaders,
)
from lib.pointnet2.models import PointNetPlusPlus, count_params
from lib.pointnet2.metrics_utils import (
    compute_iou, compute_precision_recall_f1, compute_accuracy,
)

TAG = "ablation_A2"
DESC = "A2 — 无级联管线 (单阶段 4 类 PointNet++)"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, TAG)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = 120
LR = 1e-3


class SingleStageModel(PointNetPlusPlus):
    """单阶段 4 类 — 固定半径."""
    def forward(self, xyz, geo_features=None, scale=None):
        return super().forward(xyz, geo_features=None, scale=None)


@torch.no_grad()
def validate(model, loader_fn, criterion):
    from _common import CLASS_NAMES
    model.eval()
    loader = loader_fn()
    total_loss = 0.0
    all_p, all_t = [], []
    prog = Progress(
        TextColumn("[bold]val[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TimeRemainingColumn(), console=console,
    )
    ttask = prog.add_task("val", total=len(loader), loss=0)
    with torch.no_grad(), prog:
        for pts, feat, lbl, scale in loader:
            pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
            logits = model(pts, scale=None)
            loss = criterion(logits.permute(0, 2, 1), lbl)
            total_loss += loss.item()
            all_p.append(logits.cpu())
            all_t.append(lbl.cpu())
            prog.update(ttask, advance=1, loss=loss.item())
    p, t = torch.cat(all_p), torch.cat(all_t)
    ious = compute_iou(p, t, CLASS_NAMES)
    acc = compute_accuracy(p, t)
    prf = compute_precision_recall_f1(p, t, CLASS_NAMES)
    cm = (ious["混凝土"] + ious["钢筋"] + ious["套筒"]) / 3.0
    return total_loss / len(loader), ious, acc, prf, cm


def main():
    set_seed(42)
    console.rule(f"[bold]{DESC}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}")

    make_train, make_val, make_test = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4, samples_per_epoch=3000,
    )

    model = SingleStageModel(num_classes=4, in_channels=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  [dim]参数: {n_params:,}[/]")

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS)

    best_m = 0.0
    hist = {"epoch": [], "loss": [], "val_loss": [], "acc": [], "cm": []}
    # 每类 IoU
    for cls_name in CLASS_NAMES.values():
        hist[f"iou_{cls_name}"] = []
    # 每类 PRF
    for cls_name in CLASS_NAMES.values():
        for met in ["precision", "recall", "f1"]:
            hist[f"prf_{cls_name}_{met}"] = []

    for ep in range(1, EPOCHS + 1):
        if ep > 1:
            sch.step()
        model.train()
        train_loss = 0.0
        vb = 0
        prog = Progress(
            TextColumn(f"[bold]A2-ep{ep:03d}[/]"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TimeRemainingColumn(), console=console,
        )
        ttask = prog.add_task("a2", total=len(make_train()), loss=0)
        with prog:
            for pts, feat, lbl, scale in make_train():
                pts, lbl = pts.to(DEVICE), lbl.to(DEVICE)
                opt.zero_grad()
                logits = model(pts, scale=None)
                loss = criterion(logits.permute(0, 2, 1), lbl)
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
        # 保存每类 IoU
        for cls_name in CLASS_NAMES.values():
            hist[f"iou_{cls_name}"].append(ious.get(cls_name, 0.0))
        # 保存每类 PRF
        for cls_name in CLASS_NAMES.values():
            pm = prf.get(cls_name, {})
            hist[f"prf_{cls_name}_precision"].append(pm.get("precision", 0.0))
            hist[f"prf_{cls_name}_recall"].append(pm.get("recall", 0.0))
            hist[f"prf_{cls_name}_f1"].append(pm.get("f1", 0.0))
        io = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
        pr = " | ".join(f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}" for k, v in prf.items())
        console.print(f"  A2: tl={tl:.4f} vl={vl:.4f} acc={acc:.3f} mIoU={m:.3f} | [{io}] | cm={cm:.3f}")
        console.print(f"    PRF: [{pr}]")
        if m > best_m:
            best_m = m
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pth"))
            console.print(f"  [green]↑ 最佳 (mIoU={m:.3f})[/]")
        _save_history(CHECKPOINT_DIR, TAG, hist)

    # ── 测试集 ──
    from _common import CLASS_NAMES
    console.rule("[bold blue]测试集[/]")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best.pth"), map_location=DEVICE))
    tl, ious, acc, prf, cm = validate(model, make_test, criterion)
    tio = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
    console.print(f"test_loss={tl:.4f} acc={acc:.3f} mIoU={ious['mIoU']:.3f} | [{tio}] | cm={cm:.3f}")

    with open(os.path.join(CHECKPOINT_DIR, "summary.json"), "w") as f:
        json.dump({
            "best_miou": best_m, "test_miou": ious["mIoU"],
            "test_cm": cm, **_serialize_metrics(ious, prf),
        }, f, indent=2)
    console.print(f"[bold green]完成! 最佳 mIoU={best_m:.3f}[/]")


if __name__ == "__main__":
    main()
