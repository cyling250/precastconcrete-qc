"""
敏感性 S1 — 局部块半径: 改变 PATCH_RADIUS_MM, 其余与 phase1 baseline 一致.

测试半径: 100 / 150(基线) / 200 / 300 mm
只训练 S1, 验证不同球查询半径对 S1 分割效果的影响.
"""

import os, json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from lib.pointnet2 import dataset as ds_module

from _common import (
    console, NUM_POINTS, BATCH_SIZE, LR, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_BASE, set_seed, _serialize_metrics, _save_history,
    create_dataloaders,
)
from lib.pointnet2 import PointNetPlusPlus, count_params
from lib.pointnet2.metrics import (
    compute_iou, compute_precision_recall_f1, compute_accuracy,
)

S1_RADII = {
    "S1_100": {"radius": 100.0, "desc": "局部块半径 100mm"},
    "S1_150": {"radius": 150.0, "desc": "局部块半径 150mm (基线)"},
    "S1_200": {"radius": 200.0, "desc": "局部块半径 200mm"},
    "S1_300": {"radius": 300.0, "desc": "局部块半径 300mm"},
}

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


def run_single(tag: str, radius: float, desc: str) -> dict:
    set_seed(42)
    ckpt_dir = os.path.join(CHECKPOINT_BASE, f"sensitivity_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    console.rule(f"[bold]{desc}[/]")

    ds_module.PATCH_RADIUS_MM = radius

    make_train, make_val, make_test = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4, samples_per_epoch=3000,
    )

    model = S1Model(num_classes=3, in_channels=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"  [dim]参数: {n_params:,} | 半径: {radius:.0f}mm[/]")

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
            TextColumn(f"[bold]{tag}-ep{ep:03d}[/]"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TimeRemainingColumn(), console=console,
        )
        ttask = prog.add_task(tag, total=len(make_train()), loss=0)
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
        console.print(f"  {tag}: tl={tl:.4f} vl={vl:.4f} acc={acc:.3f} mIoU={m:.3f} | [{io}] | cm={cm:.3f}")
        if m > best_m:
            best_m = m
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_s1.pth"))
            console.print(f"  [green]↑ 最佳 (mIoU={m:.3f})[/]")
        _save_history(ckpt_dir, tag, hist)

    # ── 测试集 ──
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_s1.pth"), map_location=DEVICE))
    tl, ious, acc, prf, cm = validate(model, make_test, criterion)
    io = " | ".join(f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU")
    console.print(f"  test: loss={tl:.4f} acc={acc:.3f} mIoU={ious['mIoU']:.3f} | [{io}] | cm={cm:.3f}")

    summary = {
        "radius_mm": radius, "best_val_miou": best_m,
        "test_miou": ious["mIoU"], "test_cm": cm,
        **_serialize_metrics(ious, prf),
    }
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    set_seed(42)
    console.print(f"[bold]Device:[/] {DEVICE}")
    console.print("[bold]S1 — 局部块半径敏感性[/]")

    results = {}
    for tag, cfg in S1_RADII.items():
        result = run_single(tag, cfg["radius"], cfg["desc"])
        results[tag] = result

    ds_module.PATCH_RADIUS_MM = 150.0

    # ── 汇总 ──
    console.rule("[bold cyan]S1 — 局部块半径敏感性汇总[/]")
    header = f"{'半径':>8} | {'val_mIoU':>8} | {'噪声IoU':>8} | {'混凝土IoU':>8} | {'金属IoU':>8} | {'cm':>8}"
    console.print(header)
    console.print("-" * len(header))
    for tag, r in results.items():
        console.print(f"{r['radius_mm']:.0f}mm | {r['best_val_miou']:>8.3f} | "
                      f"{r.get('iou/噪声', 0):>8.3f} | {r.get('iou/混凝土', 0):>8.3f} | "
                      f"{r.get('iou/金属', 0):>8.3f} | {r.get('test_cm', 0):>8.3f}")

    with open(os.path.join(CHECKPOINT_BASE, "sensitivity_S1_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S1 完成![/]")


if __name__ == "__main__":
    main()
