"""
敏感性 S2 — S2 体素降采样大小: 改变 VOXEL_MM, 其余与 phase2 baseline 一致.

测试体素: 20mm / 40mm(基线) / 80mm
S1 固定用 baseline checkpoint, 只重新训练 S2.
验证体素大小对钢筋/套筒 IoU 的影响.
"""

import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from _common import (
    console, NUM_POINTS, BATCH_SIZE, LR, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_BASE, set_seed, _save_history,
    create_dataloaders, CLASS_NAMES,
)
from experiment_phase1_baseline import make_s1_model
from lib.pointnet2 import PointNetPlusPlus, count_params
from lib.pointnet2.metrics import compute_iou, compute_precision_recall_f1, compute_accuracy

S2_NAMES = {0: "噪声(S2)", 1: "混凝土", 2: "钢筋(S2)", 3: "套筒(S2)"}
S1_CKPT_DIR = os.path.join(CHECKPOINT_BASE, "phase1_baseline")

# 模块级体素大小变量 (在 run_voxel 中设置, 被 _filter_patch 和 val_v2 读取)
_CURRENT_VOXEL_MM = 40.0

VOXEL_CONFIGS = {
    "voxel_20mm": {"voxel_mm": 20.0, "desc": "S2 体素 20mm"},
    "voxel_40mm": {"voxel_mm": 40.0, "desc": "S2 体素 40mm (基线)"},
    "voxel_80mm": {"voxel_mm": 80.0, "desc": "S2 体素 80mm"},
}

EPOCHS_S2 = 60


class Stage2Model(PointNetPlusPlus):
    def __init__(self, in_channels=6):
        super().__init__(num_classes=4, in_channels=in_channels)
    def forward(self, xyz, s1_probs, scale=None):
        return super().forward(xyz, geo_features=s1_probs, scale=None)


def _filter_patch(pts_norm, s1_probs, gt_labels, scale_mm):
    N = len(gt_labels)
    s1_pred = s1_probs.argmax(dim=-1)
    keep = torch.ones(N, dtype=torch.bool, device=pts_norm.device)
    is_conc = s1_pred == 1
    if is_conc.any():
        conc_idx = torch.where(is_conc)[0]
        conc_pts = pts_norm[conc_idx]
        vn = _CURRENT_VOXEL_MM / scale_mm
        vi = torch.floor(conc_pts / vn).long()
        sft = vi - vi.min(dim=0).values
        mv = sft.max(dim=0).values + 1
        uid = sft[:, 0] * mv[1] * mv[2] + sft[:, 1] * mv[2] + sft[:, 2]
        su, si = torch.sort(uid)
        fi = torch.ones(len(su), dtype=torch.bool, device=pts_norm.device)
        fi[1:] = su[1:] != su[:-1]
        drop = torch.ones(len(conc_idx), dtype=torch.bool, device=pts_norm.device)
        drop[si[fi]] = False
        keep[conc_idx[drop]] = False
    return pts_norm[keep], s1_probs[keep], gt_labels[keep]


def _build_s2_dataset(s1_model, loader_fn, cache_path):
    if os.path.exists(cache_path):
        console.print(f"  [dim]缓存已存在: {cache_path}[/]")
        return
    loader = loader_fn()
    samples = []
    prog = Progress(
        TextColumn("[bold]S2 预计算[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(), console=console,
    )
    task = prog.add_task("precompute", total=len(loader))
    with torch.no_grad(), prog:
        for pts, feat, lbl, scale in loader:
            pts, lbl, scale = pts.to(DEVICE), lbl.to(DEVICE), scale.to(DEVICE)
            B = pts.shape[0]
            for b in range(B):
                p, l, s = pts[b], lbl[b], scale[b].item()
                logits = s1_model(p.unsqueeze(0), scale=None)
                s1_probs = F.softmax(logits, dim=-1).squeeze(0)
                kp, ks, kl = _filter_patch(p, s1_probs, l, s)
                if len(kp) < 32:
                    continue
                samples.append((kp.cpu(), ks.cpu(), kl.cpu(), s))
            prog.update(task, advance=1)
    console.print(f"  [green]S2 数据集: {len(samples)} patches[/]")
    torch.save(samples, cache_path)


def _load_s2_loader(cache_path, batch_size, shuffle=False):
    samples = torch.load(cache_path, weights_only=False)
    class _S2Dataset(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            pts, s1p, lbl, scale = self.data[idx]
            return pts, s1p, lbl, scale
    def collate_fn(batch):
        max_n = max(b[0].shape[0] for b in batch)
        B = len(batch)
        device = batch[0][0].device
        pts_pad = torch.zeros(B, max_n, 3, device=device)
        s1p_pad = torch.zeros(B, max_n, 3, device=device)
        lbl_pad = torch.full((B, max_n), -1, dtype=torch.long, device=device)
        scale_bt = torch.zeros(B, device=device)
        for i, (p, s, l, sc) in enumerate(batch):
            n = p.shape[0]
            pts_pad[i, :n] = p.to(device)
            s1p_pad[i, :n] = s.to(device)
            lbl_pad[i, :n] = l.to(device)
            scale_bt[i] = sc
        return pts_pad, s1p_pad, lbl_pad, scale_bt
    dataset = _S2Dataset(samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)


@torch.no_grad()
def val_v2(s1_model, s2_model, loader_fn):
    s1_model.eval()
    s2_model.eval()
    loader = loader_fn()
    all_cp, all_ct = [], []
    for pts, feat, lbl, scale in loader:
        pts, lbl, scale = pts.to(DEVICE), lbl.to(DEVICE), scale.to(DEVICE)
        B = pts.shape[0]
        s1_logits = s1_model(pts, scale=None)
        s1_probs = F.softmax(s1_logits, dim=-1)
        f_pts, f_probs, f_lbl, f_s1pred, f_scale = [], [], [], [], []
        for b in range(B):
            p, l, s = pts[b], lbl[b], scale[b].item()
            spb = s1_probs[b]
            kp, ks, kl = _filter_patch(p, spb, l, s)  # uses _CURRENT_VOXEL_MM
            if len(kp) < 32:
                continue
            f_pts.append(kp.cpu())
            f_probs.append(ks.cpu())
            f_lbl.append(kl.cpu())
            f_s1pred.append(ks.argmax(dim=-1).cpu())
            f_scale.append(s)
        if not f_pts:
            continue
        max_n = max(p.shape[0] for p in f_pts)
        nb = len(f_pts)
        pad_pts = torch.zeros(nb, max_n, 3, device=DEVICE)
        pad_pr = torch.zeros(nb, max_n, 3, device=DEVICE)
        for i in range(nb):
            n = f_pts[i].shape[0]
            pad_pts[i, :n] = f_pts[i].to(DEVICE)
            pad_pr[i, :n] = f_probs[i].to(DEVICE)
        s2_logits = s2_model(pad_pts, pad_pr, scale=None)
        for i in range(nb):
            n = f_pts[i].shape[0]
            s2_pred = s2_logits[i, :n].argmax(dim=-1)
            kl = f_lbl[i].to(DEVICE)
            combined = s2_pred.clone()
            s1_conc = (f_s1pred[i].to(DEVICE) == 1)
            combined[s1_conc] = 1
            all_cp.append(combined.cpu())
            all_ct.append(kl.cpu())
    if not all_cp:
        return {k: 0.0 for k in list(S2_NAMES.values()) + ["mIoU"]}, 0.0, 0.0, 0.0
    cp, ct = torch.cat(all_cp), torch.cat(all_ct)
    ious = compute_iou(cp, ct, CLASS_NAMES)
    acc = compute_accuracy(cp, ct)
    rs_iou = (((cp == 2) & (ct == 2)).sum() + ((cp == 3) & (ct == 3)).sum()).float() / \
             (((cp == 2) | (ct == 2)).sum() + ((cp == 3) | (ct == 3)).sum()).clamp(min=1e-10)
    cm = (ious.get("混凝土", 0) + ious.get("钢筋", 0) + ious.get("套筒", 0)) / 3.0
    return ious, acc, rs_iou.item(), cm


def run_voxel(tag: str, voxel_mm: float, desc: str):
    ckpt_dir = os.path.join(CHECKPOINT_BASE, f"sensitivity_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_cache = os.path.join(ckpt_dir, "s2_train.pt")
    val_cache = os.path.join(ckpt_dir, "s2_val.pt")

    console.rule(f"[bold]{desc}[/]")

    global _CURRENT_VOXEL_MM
    _CURRENT_VOXEL_MM = voxel_mm

    # S1 (固定)
    s1_ckpt = os.path.join(S1_CKPT_DIR, "best_s1.pth")
    if not os.path.exists(s1_ckpt):
        console.print(f"[red]未找到 S1 权重 {s1_ckpt}[/]")
        return None
    s1_model = make_s1_model("pointnet++").to(DEVICE)
    s1_model.load_state_dict(torch.load(s1_ckpt, map_location=DEVICE))
    s1_model.eval()
    for p in s1_model.parameters():
        p.requires_grad = False

    make_train, make_val, make_test = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4, samples_per_epoch=3000,
    )

    # 预计算
    _build_s2_dataset(s1_model, make_train, train_cache)
    _build_s2_dataset(s1_model, make_val, val_cache)
    train_loader = _load_s2_loader(train_cache, BATCH_SIZE, shuffle=True)
    val_loader = _load_s2_loader(val_cache, BATCH_SIZE * 2, shuffle=False)

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
        ious, acc, rs_iou, cm = val_v2(s1_model, s2_model, make_val)
        console.print(f"  {tag} ep{ep:03d}: tl={s2_tl:.4f} acc={acc:.3f} mIoU={ious.get('mIoU', 0):.3f} cm={cm:.3f} rsIoU={rs_iou:.3f}")
        if cm > best_cm:
            best_cm = cm
            torch.save(s2_model.state_dict(), os.path.join(ckpt_dir, "best_s2.pth"))

    # 测试
    s2_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_s2.pth")))
    ious, acc, rs_iou, cm = val_v2(s1_model, s2_model, make_test)
    console.print(f"  [bold]test: acc={acc:.3f} mIoU={ious.get('mIoU', 0):.3f} cm={cm:.3f} rsIoU={rs_iou:.3f}[/]")
    return {"voxel_mm": voxel_mm, "test_miou": ious.get("mIoU", 0), "test_cm": cm, "test_rs_iou": rs_iou}


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
