"""
Phase 2: S2 四类精分割 (噪声/混凝土/钢筋/套筒).

管线: S1 → 混凝土4cm体素降采样 → S2

S2 输入: XYZ(3) + S1概率(3) = 6通道
S2 训练: 不训练混凝土 (ignore_index=1), 只修正噪声/钢筋/套筒
S2 推理: 固定混凝土为输入标签, S2 仅输出噪声/钢筋/套筒

训练: 全程使用 S1 预测 (无教师强制), 预计算 + 缓存
验证 V2: S1 管线 → S2 管线, 组合指标评估
"""

import os, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from _common import (
    console, NUM_POINTS, BATCH_SIZE, LR, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_BASE, set_seed, _serialize_metrics, _save_history,
    create_dataloaders, CLASS_NAMES,
)
from experiment_phase1 import FixedRadiusStage1
from lib.pointnet2 import PointNetPlusPlus, count_params
from lib.pointnet2.metrics import compute_iou, compute_precision_recall_f1, compute_accuracy

TAG = "phase2"
DESC = "Phase 2: S2 (噪声/混凝土/钢筋/套筒, S1前馈)"
CKPT_DIR = os.path.join(CHECKPOINT_BASE, TAG)
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = 80
VOXEL_MM = 40.0       # 4cm 体素 (mm)
S1_CKPT = os.path.join(CHECKPOINT_BASE, "phase1", "best_s1.pth")
TRAIN_CACHE = os.path.join(CKPT_DIR, "s2_train.pt")
VAL_CACHE = os.path.join(CKPT_DIR, "s2_val.pt")

S2_NAMES = {0: "噪声(S2)", 1: "混凝土", 2: "钢筋(S2)", 3: "套筒(S2)"}


# ═══════════════════════════════════════════════════════════
#  S2 模型: PointNet++ 6通道输入 (XYZ + S1概率)
# ═══════════════════════════════════════════════════════════

class Stage2Model(PointNetPlusPlus):
    """S2: 输入 XYZ(3) + S1_probs(3), 输出 (B,N,4)."""
    def __init__(self):
        super().__init__(num_classes=4, in_channels=6)

    def forward(self, xyz, s1_probs, scale=None):
        """xyz: (B,N,3), s1_probs: (B,N,3)"""
        return super().forward(xyz, geo_features=s1_probs, scale=None)


# ═══════════════════════════════════════════════════════════
#  体素降采样 (4cm, 仅混凝土)
# ═══════════════════════════════════════════════════════════

def _filter_patch(pts_norm, s1_probs, gt_labels, scale_mm):
    """S1 预测 → 混凝土4cm降采样 + 保留全部噪声/金属.

    Args:
        pts_norm:  (N, 3) 归一化坐标
        s1_probs:  (N, 3) S1 softmax 概率
        gt_labels: (N,)   GT 4 类标签 (0/1/2/3)
        scale_mm:  float  原始尺度
    Returns:
        kept_pts:   (M, 3) 保留的坐标
        kept_probs: (M, 3) 保留的 S1 概率
        kept_gt:    (M,)   保留的 GT 标签
    """
    N = len(gt_labels)
    s1_pred = s1_probs.argmax(dim=-1)  # (N,) {0=噪, 1=砼, 2=金}

    keep = torch.ones(N, dtype=torch.bool, device=pts_norm.device)

    # 对 S1 预测为混凝土的点做 4cm 体素降采样
    is_conc = s1_pred == 1
    if is_conc.any():
        conc_idx = torch.where(is_conc)[0]
        conc_pts = pts_norm[conc_idx]
        vn = VOXEL_MM / scale_mm
        vi = torch.floor(conc_pts / vn).long()
        sft = vi - vi.min(dim=0).values
        mv = sft.max(dim=0).values + 1
        uid = sft[:, 0] * mv[1] * mv[2] + sft[:, 1] * mv[2] + sft[:, 2]
        su, si = torch.sort(uid)
        fi = torch.ones(len(su), dtype=torch.bool, device=pts_norm.device)
        fi[1:] = su[1:] != su[:-1]
        # 不保留降采样以外的混凝土
        drop = torch.ones(len(conc_idx), dtype=torch.bool, device=pts_norm.device)
        drop[si[fi]] = False
        keep[conc_idx[drop]] = False

    return pts_norm[keep], s1_probs[keep], gt_labels[keep]


# ═══════════════════════════════════════════════════════════
#  预计算 S1 + 缓存
# ═══════════════════════════════════════════════════════════

def _build_s2_dataset(s1_model, loader_fn, cache_path):
    """用 S1 预计算全部 patch → 过滤 → 缓存到磁盘."""
    if os.path.exists(cache_path):
        console.print(f"  [dim]S2 缓存已存在: {cache_path}[/]")
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
                # S1 推理
                logits = s1_model(p.unsqueeze(0), scale=None)  # (1, N, 3)
                s1_probs = F.softmax(logits, dim=-1).squeeze(0)  # (N, 3)
                # 过滤 (混凝土降采样)
                kp, ks, kl = _filter_patch(p, s1_probs, l, s)
                if len(kp) < 32:
                    continue
                samples.append((kp.cpu(), ks.cpu(), kl.cpu(), s))
            prog.update(task, advance=1)
    console.print(f"  [green]S2 数据集: {len(samples)} patches[/]")
    torch.save(samples, cache_path)
    console.print(f"  [dim]缓存已保存: {cache_path}[/]")


def _load_s2_loader(cache_path, batch_size, shuffle=False):
    """从缓存加载 S2 数据集, 返回 DataLoader (变长 padding)."""
    samples = torch.load(cache_path, weights_only=False)

    class _S2Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            pts, s1p, lbl, scale = self.data[idx]
            return pts, s1p, lbl, scale

    def collate_fn(batch):
        # batch: list of (pts, s1p, lbl, scale)
        max_n = max(b[0].shape[0] for b in batch)
        B = len(batch)
        device = batch[0][0].device

        pts_pad = torch.zeros(B, max_n, 3, device=device)
        s1p_pad = torch.zeros(B, max_n, 3, device=device)
        lbl_pad = torch.full((B, max_n), -1, dtype=torch.long, device=device)  # -1 = padding
        scale_bt = torch.zeros(B, device=device)

        for i, (p, s, l, sc) in enumerate(batch):
            n = p.shape[0]
            pts_pad[i, :n] = p.to(device)
            s1p_pad[i, :n] = s.to(device)
            lbl_pad[i, :n] = l.to(device)
            scale_bt[i] = sc

        return pts_pad, s1p_pad, lbl_pad, scale_bt

    dataset = _S2Dataset(samples)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0,
    )


# ═══════════════════════════════════════════════════════════
#  V2 验证 (在保留点上直接计算)
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def val_v2_simple(s1_model, s2_model, loader_fn):
    """V2 验证: batch S1+S2 推理加速.

    组合预测: S2_pred[非混凝土] + 输入混凝土标签(1)
    """
    s1_model.eval()
    s2_model.eval()
    loader = loader_fn()

    all_sp, all_st = [], []           # S2 纯预测 vs GT (仅非混凝土)
    all_cp, all_ct = [], []           # 组合预测 vs GT (全部保留点)

    prog = Progress(
        TextColumn("[bold]V2[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(), console=console,
    )
    task = prog.add_task("v2", total=len(loader))

    with torch.no_grad(), prog:
        for pts, feat, lbl, scale in loader:
            pts, lbl, scale = pts.to(DEVICE), lbl.to(DEVICE), scale.to(DEVICE)
            B = pts.shape[0]

            # ── Batch S1 推理 ──
            s1_logits = s1_model(pts, scale=None)  # (B, N, 3)
            s1_probs = F.softmax(s1_logits, dim=-1)  # (B, N, 3)

            # ── 逐样本过滤 + 收集 S2 输入 ──
            f_pts, f_probs, f_lbl, f_s1pred, f_scale = [], [], [], [], []
            for b in range(B):
                p, l, s = pts[b], lbl[b], scale[b].item()
                spb = s1_probs[b]
                kp, ks, kl = _filter_patch(p, spb, l, s)
                if len(kp) < 32:
                    continue
                f_pts.append(kp.cpu())
                f_probs.append(ks.cpu())
                f_lbl.append(kl.cpu())
                f_s1pred.append(ks.argmax(dim=-1).cpu())  # S1 argmax on kept
                f_scale.append(s)

            if not f_pts:
                prog.update(task, advance=1)
                continue

            # ── Batch S2 推理 (padding) ──
            max_n = max(p.shape[0] for p in f_pts)
            nb = len(f_pts)
            pad_pts = torch.zeros(nb, max_n, 3, device=DEVICE)
            pad_pr = torch.zeros(nb, max_n, 3, device=DEVICE)
            pad_sc = torch.zeros(nb, device=DEVICE)
            for i in range(nb):
                n = f_pts[i].shape[0]
                pad_pts[i, :n] = f_pts[i].to(DEVICE)
                pad_pr[i, :n] = f_probs[i].to(DEVICE)
                pad_sc[i] = f_scale[i]

            s2_logits = s2_model(pad_pts, pad_pr, scale=pad_sc)  # (nb, max_n, 4)

            # ── 逐样本收集指标 ──
            for i in range(nb):
                n = f_pts[i].shape[0]
                s2_pred = s2_logits[i, :n].argmax(dim=-1)  # (M,)
                kl = f_lbl[i].to(DEVICE)

                # 组合预测: S1 预测为混凝土的点保持为 1, 其余用 S2
                combined = s2_pred.clone()
                s1_conc = (f_s1pred[i].to(DEVICE) == 1)
                combined[s1_conc] = 1

                all_cp.append(combined.cpu())
                all_ct.append(kl.cpu())

                # S2 纯预测 (排除混凝土)
                non_conc = kl != 1
                if non_conc.any():
                    all_sp.append(s2_pred[non_conc].cpu())
                    all_st.append(kl[non_conc].cpu())

            prog.update(task, advance=1)

    if not all_cp:
        z = {k: 0.0 for k in list(S2_NAMES.values()) + ["mIoU"]}
        z2 = {k: {"precision": 0, "recall": 0, "f1": 0} for k in S2_NAMES.values()}
        return z, z2, 0.0, z, z2, 0.0, 0.0

    # ── 组合指标 (4 类) ──
    cp = torch.cat(all_cp)
    ct = torch.cat(all_ct)
    c_ious = compute_iou(cp, ct, CLASS_NAMES)
    c_acc = compute_accuracy(cp, ct)
    c_prf = compute_precision_recall_f1(cp, ct, CLASS_NAMES)

    # ── S2 纯指标 (噪声/钢筋/套筒) ──
    if all_sp:
        sp = torch.cat(all_sp)
        st = torch.cat(all_st)
        s2_names_sub = {0: "噪声(S2)", 2: "钢筋(S2)", 3: "套筒(S2)"}
        s_ious = {}
        for cls_id, cls_name in [(0, "噪声(S2)"), (2, "钢筋(S2)"), (3, "套筒(S2)")]:
            mask_t = (st == cls_id)
            mask_p = (sp == cls_id)
            inter = (mask_t & mask_p).sum().float()
            union = (mask_t | mask_p).sum().float()
            s_ious[cls_name] = (inter / (union + 1e-10)).item()
        s_ious["mIoU"] = sum(s_ious.get(n, 0) for n in s2_names_sub.values()) / 3.0
    else:
        s_ious = {"mIoU": 0.0}
        for n in ["噪声(S2)", "钢筋(S2)", "套筒(S2)"]:
            s_ious[n] = 0.0

    # ── 钢筋&套筒 IoU ──
    rs_inter = ((cp == 2) & (ct == 2)).sum().float() + ((cp == 3) & (ct == 3)).sum().float()
    rs_union = ((cp == 2) | (ct == 2)).sum().float() + ((cp == 3) | (ct == 3)).sum().float()
    rs_iou = (rs_inter / (rs_union + 1e-10)).item()

    # ── 干净 mIoU (混凝土+钢筋+套筒) ──
    cm = (c_ious.get("混凝土", 0) + c_ious.get("钢筋", 0) + c_ious.get("套筒", 0)) / 3.0

    return c_ious, c_prf, c_acc, s_ious, rs_iou, cm


def print_v2(ep, tl, c_ious, c_prf, c_acc, s_ious, rs_iou, cm):
    """打印 V2 指标."""
    cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
    cpr = " | ".join(f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}" for k, v in c_prf.items())
    console.print(f"  S2: tl={tl:.4f} acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}")
    console.print(f"    PRF(组合): [{cpr}]")
    sio = " | ".join(f"{k}={v:.3f}" for k, v in s_ious.items() if k != "mIoU")
    console.print(f"    S2纯: [{sio}] mIoU(S2)={s_ious.get('mIoU', 0):.3f}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    set_seed(42)
    console.rule(f"[bold]{DESC}[/]")
    console.print(f"[bold]Device:[/] {DEVICE}")
    console.print(f"  体素: {VOXEL_MM}mm (4cm), 仅混凝土降采样, 保留全部噪声+金属")

    # ── 检查 S1 权重 ──
    if not os.path.exists(S1_CKPT):
        console.print(f"[red]未找到 S1 权重 {S1_CKPT}, 请先运行 experiment_phase1.py[/]")
        return

    # ── S1 模型 (冻结, 仅推理) ──
    s1_model = FixedRadiusStage1(num_classes=3, in_channels=3).to(DEVICE)
    s1_model.load_state_dict(torch.load(S1_CKPT, map_location=DEVICE))
    s1_model.eval()
    for p in s1_model.parameters():
        p.requires_grad = False
    console.print(f"  [dim]S1 已加载 (冻结)[/]")

    # ── 数据加载 ──
    make_train, make_val, make_test = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4, samples_per_epoch=3000,
    )

    # ── 预计算 S2 数据集 ──
    console.rule("[bold]预计算 S2 数据集[/]")
    _build_s2_dataset(s1_model, make_train, TRAIN_CACHE)
    _build_s2_dataset(s1_model, make_val, VAL_CACHE)

    train_loader = _load_s2_loader(TRAIN_CACHE, BATCH_SIZE, shuffle=True)
    val_loader = _load_s2_loader(VAL_CACHE, BATCH_SIZE * 2, shuffle=False)

    n_train = len(torch.load(TRAIN_CACHE, weights_only=False))
    console.print(f"  [bold]训练:[/] {n_train} patches ({len(train_loader)} batch/epoch)")
    console.print(f"  [bold]验证:[/] {len(torch.load(VAL_CACHE, weights_only=False))} patches ({len(val_loader)} batch)")

    # ── S2 模型 ──
    s2_model = Stage2Model().to(DEVICE)
    n_p = sum(p.numel() for p in s2_model.parameters())
    console.print(f"  [dim]S2 参数: {n_p:,}[/]")

    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # -1 = padding
    opt = torch.optim.AdamW(s2_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS)

    best_cm = 0.0
    hist = {
        "epoch": [], "loss": [], "val_loss": [],
        "val_acc": [], "val_cm": [], "val_rs_iou": [],
    }

    for ep in range(1, EPOCHS + 1):
        if ep > 1:
            sch.step()
        # ── 训练 ──
        s2_model.train()
        tl = 0.0
        n_b = 0
        prog = Progress(
            TextColumn(f"[bold]S2-ep{ep:03d}[/]"), BarColumn(),
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

                # S2 前向
                logits = s2_model(pts_pad, s1p_pad, scale=scale_bt)  # (B, M, 4)

                # 损失: 混凝土(1)和padding(-1)都映射到-1忽略
                loss_lbl = lbl_pad.clone()
                loss_lbl[loss_lbl == 1] = -1  # 混凝土不参与梯度
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

        # ── 验证 V2 ──
        c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
            s1_model, s2_model, make_val,
        )
        print_v2(ep, s2_tl, c_ious, c_prf, c_acc, s_ious, rs_iou, cm)

        hist["epoch"].append(ep)
        hist["loss"].append(s2_tl)
        hist["val_loss"].append(c_ious.get("mIoU", 0))
        hist["val_acc"].append(c_acc)
        hist["val_cm"].append(cm)
        hist["val_rs_iou"].append(rs_iou)

        if cm > best_cm:
            best_cm = cm
            torch.save({
                "epoch": ep,
                "model_state": s2_model.state_dict(),
                "opt_state": opt.state_dict(),
                "cm": cm,
                "c_ious": c_ious,
                "rs_iou": rs_iou,
            }, os.path.join(CKPT_DIR, "best_s2.pth"))
            console.print(f"  [green]↑ 最佳 (cm={cm:.3f})[/]")

        _save_history(CKPT_DIR, TAG, hist)

    # ── 测试集 ──
    console.rule("[bold blue]测试集 (S1→S2 全管线)[/]")
    ckpt = os.path.join(CKPT_DIR, "best_s2.pth")
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=DEVICE)
        s2_model.load_state_dict(sd["model_state"])
        c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(
            s1_model, s2_model, make_test,
        )
        cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
        console.print(f"  test_acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}")
    else:
        console.print("[yellow]未找到 best_s2.pth[/]")

    with open(os.path.join(CKPT_DIR, "summary_phase2.json"), "w") as f:
        json.dump({
            "best_cm": best_cm,
            "test_miou": c_ious.get("mIoU", 0),
            "test_cm": cm,
            "test_rs_iou": rs_iou,
        }, f, indent=2)
    console.print(f"[bold green]完成! 最佳 cm={best_cm:.3f}[/]")


if __name__ == "__main__":
    main()
