"""
Shared cascade pipeline utilities: S1 → adaptive voxel downsampling → S2.

All functions accept voxel_mm as a parameter for reuse by both the baseline
and sensitivity experiments (e.g., voxel size sweep).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..models import PointNetPlusPlus, count_params
from ..metrics_utils import compute_iou, compute_precision_recall_f1, compute_accuracy

# ── Constants ──
S2_NAMES = {0: "噪声(S2)", 1: "混凝土", 2: "钢筋(S2)", 3: "套筒(S2)"}


# ═══════════════════════════════════════════════════════════
#  S2 模型: PointNet++ 6通道输入 (XYZ + S1概率)
# ═══════════════════════════════════════════════════════════


class Stage2Model(PointNetPlusPlus):
    """S2: 输入 XYZ(3) + S1_probs(3), 输出 (B,N,4)."""
    def __init__(self, in_channels=6):
        super().__init__(num_classes=4, in_channels=in_channels)

    def forward(self, xyz, s1_probs, scale=None):
        """xyz: (B,N,3), s1_probs: (B,N,3)"""
        return super().forward(xyz, geo_features=s1_probs, scale=None)


# ═══════════════════════════════════════════════════════════
#  体素降采样 (仅混凝土)
# ═══════════════════════════════════════════════════════════


def filter_patch(pts_norm, s1_probs, gt_labels, scale_mm, voxel_mm=40.0,
                 teacher_forcing=False):
    """S1 预测 → 混凝土 voxel_mm 降采样 + 保留全部噪声/金属.

    Args:
        pts_norm: (N, 3) normalized points
        s1_probs: (N, 3) S1 softmax probabilities
        gt_labels: (N,) ground truth labels
        scale_mm: float, normalization scale in mm
        voxel_mm: float, voxel size in mm (default 40mm = 4cm)
        teacher_forcing: if True, use GT labels to identify concrete
                         (not S1 predictions) for voxel downsampling
    Returns:
        filtered_pts, filtered_probs, filtered_labels
    """
    N = len(gt_labels)
    s1_pred = s1_probs.argmax(dim=-1)  # (N,) {0=噪, 1=砼, 2=金}

    keep = torch.ones(N, dtype=torch.bool, device=pts_norm.device)

    # 教师强制: 用 GT 标签识别混凝土; 否则用 S1 预测
    if teacher_forcing:
        # GT: 0=噪声, 1=混凝土, 2=钢筋, 3=套筒
        is_conc = gt_labels == 1
    else:
        is_conc = s1_pred == 1

    if is_conc.any():
        conc_idx = torch.where(is_conc)[0]
        conc_pts = pts_norm[conc_idx]
        vn = voxel_mm / scale_mm
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


def build_s2_dataset(s1_model, loader_fn, cache_path, device, console,
                      voxel_mm=40.0, teacher_forcing=False):
    """用 S1 预计算全部 patch → 过滤 → 缓存到磁盘.

    Args:
        teacher_forcing: if True, use GT labels for concrete identification
                         (saved with _tf suffix in cache name)
    """
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

    # 教师强制数据集用不同缓存名
    actual_path = cache_path
    if teacher_forcing:
        base, ext = os.path.splitext(cache_path)
        actual_path = f"{base}_tf{ext}"

    if os.path.exists(actual_path):
        mode = "教师强制" if teacher_forcing else "S1预测"
        console.print(f"  [dim]S2 缓存已存在 ({mode}): {actual_path}[/]")
        return

    mode_tag = "TF" if teacher_forcing else "S1"
    loader = loader_fn()
    samples = []
    prog = Progress(
        TextColumn(f"[bold]S2[{mode_tag}] 预计算[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(), console=console,
    )
    task = prog.add_task("precompute", total=len(loader))
    with torch.no_grad(), prog:
        for pts, feat, lbl, scale in loader:
            pts, lbl, scale = pts.to(device), lbl.to(device), scale.to(device)
            B = pts.shape[0]
            # 整 batch S1 推理 (比逐样本快 ~16 倍)
            logits = s1_model(pts, scale=None)  # (B, N, 3)
            s1_probs = F.softmax(logits, dim=-1)  # (B, N, 3)
            for b in range(B):
                p, l, s = pts[b], lbl[b], scale[b].item()
                spb = s1_probs[b]  # (N, 3)
                # 过滤 (混凝土降采样, 支持教师强制)
                kp, ks, kl = filter_patch(p, spb, l, s,
                                          voxel_mm=voxel_mm,
                                          teacher_forcing=teacher_forcing)
                if len(kp) < 32:
                    continue
                samples.append((kp.cpu(), ks.cpu(), kl.cpu(), s))
            prog.update(task, advance=1)
    mode_name = "教师强制" if teacher_forcing else "S1预测"
    console.print(f"  [green]S2 数据集 ({mode_name}): {len(samples)} patches[/]")
    torch.save(samples, actual_path)
    console.print(f"  [dim]缓存已保存: {actual_path}[/]")


def load_s2_loader(cache_path, batch_size, shuffle=False, teacher_forcing=False):
    """从缓存加载 S2 数据集, 返回 DataLoader (变长 padding).

    Args:
        teacher_forcing: if True, load the _tf variant of the cache
    """
    actual_path = cache_path
    if teacher_forcing:
        base, ext = os.path.splitext(cache_path)
        actual_path = f"{base}_tf{ext}"
    samples = torch.load(actual_path, weights_only=False)

    class _S2Dataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
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
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0,
    )


# ═══════════════════════════════════════════════════════════
#  V2 验证
# ═══════════════════════════════════════════════════════════


@torch.no_grad()
def val_v2_simple(s1_model, s2_model, loader_fn, device, class_names,
                   voxel_mm=40.0, teacher_forcing=False):
    """V2 验证: batch S1+S2 推理加速.

    Args:
        teacher_forcing: if True, use GT labels for concrete filtering
                         (evaluates S2's refinement on clean concrete masks)
    """
    s1_model.eval()
    s2_model.eval()
    loader = loader_fn()

    all_sp, all_st = [], []
    all_cp, all_ct = [], []

    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from ..metrics_utils import compute_accuracy

    prog = Progress(
        TextColumn("[bold]V2[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(), console=None,
    )

    with torch.no_grad():
        for pts, feat, lbl, scale in loader:
            pts, lbl, scale = pts.to(device), lbl.to(device), scale.to(device)
            B = pts.shape[0]

            s1_logits = s1_model(pts, scale=None)
            s1_probs = F.softmax(s1_logits, dim=-1)

            f_pts, f_probs, f_lbl, f_s1pred, f_scale = [], [], [], [], []
            for b in range(B):
                p, l, s = pts[b], lbl[b], scale[b].item()
                spb = s1_probs[b]
                kp, ks, kl = filter_patch(p, spb, l, s, voxel_mm=voxel_mm,
                                          teacher_forcing=teacher_forcing)
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
            pad_pts = torch.zeros(nb, max_n, 3, device=device)
            pad_pr = torch.zeros(nb, max_n, 3, device=device)
            pad_sc = torch.zeros(nb, device=device)
            for i in range(nb):
                n = f_pts[i].shape[0]
                pad_pts[i, :n] = f_pts[i].to(device)
                pad_pr[i, :n] = f_probs[i].to(device)
                pad_sc[i] = f_scale[i]

            s2_logits = s2_model(pad_pts, pad_pr, scale=pad_sc)

            for i in range(nb):
                n = f_pts[i].shape[0]
                s2_pred = s2_logits[i, :n].argmax(dim=-1)
                kl = f_lbl[i].to(device)

                combined = s2_pred.clone()
                s1_conc = (f_s1pred[i].to(device) == 1)
                combined[s1_conc] = 1

                all_cp.append(combined.cpu())
                all_ct.append(kl.cpu())

                non_conc = kl != 1
                if non_conc.any():
                    all_sp.append(s2_pred[non_conc].cpu())
                    all_st.append(kl[non_conc].cpu())

    if not all_cp:
        z = {k: 0.0 for k in list(class_names.values()) + ["mIoU"]}
        z2 = {k: {"precision": 0, "recall": 0, "f1": 0} for k in class_names.values()}
        return z, z2, 0.0, z, z2, 0.0, 0.0

    cp = torch.cat(all_cp)
    ct = torch.cat(all_ct)
    c_ious = compute_iou(cp, ct, class_names)
    c_acc = compute_accuracy(cp, ct)
    c_prf = compute_precision_recall_f1(cp, ct, class_names)

    if all_sp:
        sp = torch.cat(all_sp)
        st = torch.cat(all_st)
        s_ious = {}
        for cls_id, cls_name in [(0, "噪声(S2)"), (2, "钢筋(S2)"), (3, "套筒(S2)")]:
            mask_t = (st == cls_id)
            mask_p = (sp == cls_id)
            inter = (mask_t & mask_p).sum().float()
            union = (mask_t | mask_p).sum().float()
            s_ious[cls_name] = (inter / (union + 1e-10)).item()
        s_ious["mIoU"] = sum(s_ious.get(n, 0) for n in ["噪声(S2)", "钢筋(S2)", "套筒(S2)"]) / 3.0
    else:
        s_ious = {"mIoU": 0.0}
        for n in ["噪声(S2)", "钢筋(S2)", "套筒(S2)"]:
            s_ious[n] = 0.0

    rs_inter = ((cp == 2) & (ct == 2)).sum().float() + ((cp == 3) & (ct == 3)).sum().float()
    rs_union = ((cp == 2) | (ct == 2)).sum().float() + ((cp == 3) | (ct == 3)).sum().float()
    rs_iou = (rs_inter / (rs_union + 1e-10)).item()

    cm = (c_ious.get("混凝土", 0) + c_ious.get("钢筋", 0) + c_ious.get("套筒", 0)) / 3.0

    return c_ious, c_prf, c_acc, s_ious, rs_iou, cm


def print_v2(ep, tl, c_ious, c_prf, c_acc, s_ious, rs_iou, cm, console=None):
    """Print V2 validation results."""
    if console is None:
        console = __import__("rich").console.Console()
    cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
    cpr = " | ".join(
        f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
        for k, v in c_prf.items()
    )
    console.print(f"  S2: tl={tl:.4f} acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}")
    console.print(f"    PRF(组合): [{cpr}]")
    sio = " | ".join(f"{k}={v:.3f}" for k, v in s_ious.items() if k != "mIoU")
    console.print(f"    S2纯: [{sio}] mIoU(S2)={s_ious.get('mIoU', 0):.3f}")
