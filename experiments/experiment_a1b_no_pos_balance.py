"""
消融 A1b — 无正样本平衡 (pos_ratio=1.0, 自然分布).

验证 pos_ratio=0.8 正样本平衡策略的必要性。
只训练 S1，仅改动 create_dataloaders(pos_ratio=1.0)。
"""

import os, json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from _common import (
    console, NUM_POINTS, BATCH_SIZE, LR, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_BASE, set_seed, _serialize_metrics, _save_history,
    create_dataloaders,
)
from lib.pointnet2.models import PointNetPlusPlus, count_params
from lib.pointnet2.metrics_utils import (
    compute_iou, compute_precision_recall_f1, compute_accuracy,
)

TAG = "ablation_A1b"
DESC = "A1b — 无正样本平衡 (pos_ratio=1.0)"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, TAG)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = 120
LR = 1e-3
S1_NAMES = {0: "噪声", 1: "混凝土", 2: "金属"}


class S1Model(PointNetPlusPlus):
    def forward(self, xyz, geo_features=None, scale=None):
        return super().forward(xyz, geo_features=None, scale=None)


def _map_labels(lbl):
    clbl = lbl.clone()
    clbl[clbl == 3] = 2
    return clbl


@torch.no_grad()
def validate(model, loader_fn, criterion):
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


def main():
    set_seed(42)
    console.rule(f"[bold]{DESC}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}")

    make_train, make_val, make_test = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4,
        samples_per_epoch=3000, pos_ratio=1.0,
    )

    model = S1Model(num_classes=3, in_channels=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  [dim]参数: {n_params:,}[/]")

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS)

    best_m = 0.0
    hist = {"epoch": [], "loss": [], "val_loss": [], "acc": [], "cm": []}

    for ep in range(1, EPOCHS + 1):
        if ep > 1:
            sch.step()
        model.train()
        train_loss = 0.0
        vb = 0
        prog = Progress(
            TextColumn(f"[bold]A1b-ep{ep:03d}[/]"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TimeRemainingColumn(), console=console,
        )
        ttask = prog.add_task("a1b", total=len(make_train()), loss=0)
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
        pr = " | ".join(f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}" for k, v in prf.items())
        console.print(f"  A1b: tl={tl:.4f} vl={vl:.4f} acc={acc:.3f} mIoU={m:.3f} | [{io}] | cm={cm:.3f}")
        console.print(f"    PRF: [{pr}]")
        if m > best_m:
            best_m = m
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_s1.pth"))
            console.print(f"  [green]↑ 最佳 (mIoU={m:.3f})[/]")
        _save_history(CHECKPOINT_DIR, TAG, hist)

    # ── 测试集 ──
    console.rule("[bold blue]测试集[/]")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_s1.pth"), map_location=DEVICE))
    tl, ious, acc, prf, cm = validate(model, make_test, criterion)
    tio = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
    console.print(f"test_loss={tl:.4f} acc={acc:.3f} mIoU={ious['mIoU']:.3f} | [{tio}] | cm={cm:.3f}")
    pr_str = " | ".join(f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}" for k, v in prf.items())
    console.print(f"  PRF: [{pr_str}]")

    with open(os.path.join(CHECKPOINT_DIR, "summary.json"), "w") as f:
        json.dump({"best_miou": best_m, "test_miou": ious["mIoU"], "test_cm": cm, **_serialize_metrics(ious, prf)}, f, indent=2)
    console.print(f"[bold green]完成! 最佳 mIoU={best_m:.3f}[/]")


if __name__ == "__main__":
    main()
