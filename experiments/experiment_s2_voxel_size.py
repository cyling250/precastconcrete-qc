"""
敏感性 S2 — S2 体素降采样大小: 改变 VOXEL_MM, 其余与 phase2 baseline 一致.

测试体素: 20mm / 40mm(基线) / 80mm
S1 固定用 baseline checkpoint, 只重新训练 S2.
验证体素大小对钢筋/套筒 IoU 的影响.
"""

import os, json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from _common import (
    console, NUM_POINTS, BATCH_SIZE, LR, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_BASE, set_seed, _save_history,
    create_dataloaders, CLASS_NAMES,
)
from experiment_phase1_baseline import make_s1_model
from lib.pointnet2.doublephase import (
    Stage2Model, build_s2_dataset, load_s2_loader,
    val_v2_simple,
)

S1_CKPT_DIR = os.path.join(CHECKPOINT_BASE, "phase1_baseline_pointmlp")

VOXEL_CONFIGS = {
    "voxel_20mm": {"voxel_mm": 20.0, "desc": "S2 体素 20mm"},
    "voxel_40mm": {"voxel_mm": 40.0, "desc": "S2 体素 40mm (基线)"},
    "voxel_80mm": {"voxel_mm": 80.0, "desc": "S2 体素 80mm"},
}

EPOCHS_S2 = 60


def run_voxel(tag: str, voxel_mm: float, desc: str):
    ckpt_dir = os.path.join(CHECKPOINT_BASE, f"sensitivity_{tag}_pointmlp")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_cache = os.path.join(ckpt_dir, "s2_train.pt")
    val_cache = os.path.join(ckpt_dir, "s2_val.pt")

    console.rule(f"[bold]{desc}[/]")

    # S1 (固定)
    s1_ckpt = os.path.join(S1_CKPT_DIR, "best_s1.pth")
    if not os.path.exists(s1_ckpt):
        console.print(f"[red]未找到 S1 权重 {s1_ckpt}[/]")
        return None
    s1_model = make_s1_model("pointmlp").to(DEVICE)
    s1_model.load_state_dict(torch.load(s1_ckpt, map_location=DEVICE))
    s1_model.eval()
    for p in s1_model.parameters():
        p.requires_grad = False

    make_train, make_val, make_test = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4, samples_per_epoch=3000,
    )

    # 预计算
    build_s2_dataset(s1_model, make_train, train_cache, DEVICE, console, voxel_mm=voxel_mm)
    build_s2_dataset(s1_model, make_val, val_cache, DEVICE, console, voxel_mm=voxel_mm)
    train_loader = load_s2_loader(train_cache, BATCH_SIZE, shuffle=True)
    val_loader = load_s2_loader(val_cache, BATCH_SIZE * 2, shuffle=False)

    s2_model = Stage2Model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.AdamW(s2_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS_S2)

    best_cm = 0.0
    for ep in range(1, EPOCHS_S2 + 1):
        if ep > 1:
            sch.step()
        s2_model.train()
        tl, nb = 0.0, 0
        for pts_pad, s1p_pad, lbl_pad, scale_bt in train_loader:
            pts_pad, s1p_pad, lbl_pad = pts_pad.to(DEVICE), s1p_pad.to(DEVICE), lbl_pad.to(DEVICE)
            opt.zero_grad()
            logits = s2_model(pts_pad, s1p_pad, scale=None)
            loss_lbl = lbl_pad.clone()
            loss_lbl[loss_lbl == 1] = -1
            loss = criterion(logits.permute(0, 2, 1), loss_lbl)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s2_model.parameters(), max_norm=1.0)
            opt.step()
            tl += loss.item()
            nb += 1
        s2_tl = tl / max(nb, 1)
        c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
            s1_model, s2_model, make_val, DEVICE, CLASS_NAMES, voxel_mm=voxel_mm,
        )
        console.print(f"  {tag} ep{ep:03d}: tl={s2_tl:.4f} acc={c_acc:.3f} mIoU={c_ious.get('mIoU', 0):.3f} cm={cm:.3f} rsIoU={rs_iou:.3f}")
        if cm > best_cm:
            best_cm = cm
            torch.save(s2_model.state_dict(), os.path.join(ckpt_dir, "best_s2.pth"))

    # 测试
    s2_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_s2.pth")))
    c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
        s1_model, s2_model, make_test, DEVICE, CLASS_NAMES, voxel_mm=voxel_mm,
    )
    console.print(f"  [bold]test: acc={c_acc:.3f} mIoU={c_ious.get('mIoU', 0):.3f} cm={cm:.3f} rsIoU={rs_iou:.3f}[/]")
    return {"voxel_mm": voxel_mm, "test_miou": c_ious.get("mIoU", 0), "test_cm": cm, "test_rs_iou": rs_iou}


def main():
    set_seed(42)
    console.print(f"[bold]Device:[/] {DEVICE}")
    console.print("[bold]S2 — S2 体素大小敏感性[/]")

    results = {}
    for tag, cfg in VOXEL_CONFIGS.items():
        r = run_voxel(tag, cfg["voxel_mm"], cfg["desc"])
        if r:
            results[tag] = r

    console.rule("[bold cyan]S2 — 体素大小敏感性汇总[/]")
    for tag, r in results.items():
        console.print(f"  {tag}: {r['voxel_mm']:.0f}mm | test_mIoU={r['test_miou']:.3f} cm={r['test_cm']:.3f} rsIoU={r['test_rs_iou']:.3f}")

    with open(os.path.join(CHECKPOINT_BASE, "sensitivity_S2_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[bold green]S2 完成![/]")


if __name__ == "__main__":
    main()
