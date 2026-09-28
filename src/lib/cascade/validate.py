"""
级联管线验证: V2 全管线评估 (S1→S2) + S2 缓存版评估.
"""

import torch
import torch.nn.functional as F

from lib.cascade.filter import filter_patch
from lib.utils.metrics import compute_iou, compute_precision_recall_f1, compute_accuracy, cm_iou


S2_NAMES = {0: "噪声(S2)", 1: "混凝土", 2: "钢筋(S2)", 3: "套筒(S2)"}


@torch.no_grad()
def val_v2_simple(s1_model, s2_model, loader_fn, device, class_names,
                   voxel_mm: float = 40.0):
    """V2 全管线验证: S1 实时推理 + S2 推理."""
    s1_model.eval()
    s2_model.eval()
    loader = loader_fn()

    all_sp, all_st = [], []
    all_cp, all_ct = [], []

    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.console import Console

    prog = Progress(
        TextColumn("[bold]V2[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(), console=Console(),
    )
    vtask = prog.add_task("val", total=len(loader))

    with torch.no_grad(), prog:
        for pts, feat, lbl, scale in loader:
            pts, lbl, scale = pts.to(device), lbl.to(device), scale.to(device)
            B = pts.shape[0]

            s1_logits = s1_model(pts, scale=None)
            s1_probs = F.softmax(s1_logits, dim=-1)

            f_pts, f_probs, f_lbl, f_s1pred, f_scale = [], [], [], [], []
            for b in range(B):
                p, l, s = pts[b], lbl[b], scale[b].item()
                spb = s1_probs[b]
                kp, ks, kl = filter_patch(p, spb, l, s, voxel_mm=voxel_mm)
                if len(kp) < 32:
                    continue
                f_pts.append(kp.cpu())
                f_probs.append(ks.cpu())
                f_lbl.append(kl.cpu())
                f_s1pred.append(ks.argmax(dim=-1).cpu())
                f_scale.append(s)

            if not f_pts:
                prog.update(vtask, advance=1)
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

            prog.update(vtask, advance=1)

    if not all_cp:
        return _empty_results(class_names)

    cp = torch.cat(all_cp)
    ct = torch.cat(all_ct)
    c_ious = compute_iou(cp, ct, class_names)
    c_acc = compute_accuracy(cp, ct)
    c_prf = compute_precision_recall_f1(cp, ct, class_names)

    s_ious = _compute_s2_pure_ious(all_sp, all_st)

    rs_inter = ((cp == 2) & (ct == 2)).sum().float() + ((cp == 3) & (ct == 3)).sum().float()
    rs_union = ((cp == 2) | (ct == 2)).sum().float() + ((cp == 3) | (ct == 3)).sum().float()
    rs_iou = (rs_inter / (rs_union + 1e-10)).item()

    cm = cm_iou(c_ious.get("混凝土", 0), c_ious.get("钢筋", 0), c_ious.get("套筒", 0))

    return c_ious, c_prf, c_acc, s_ious, rs_iou, cm


@torch.no_grad()
def val_s2_cached(s2_model, s2_loader, device, class_names):
    """只跑 S2 验证 (S1 已预计算缓存)."""
    s2_model.eval()

    all_sp, all_st = [], []
    all_cp, all_ct = [], []

    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.console import Console

    prog = Progress(
        TextColumn("[bold]V2-cached[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(), console=Console(),
    )
    vtask = prog.add_task("val", total=len(s2_loader))

    with torch.no_grad(), prog:
        for pts_pad, s1p_pad, lbl_pad, scale_bt in s2_loader:
            pts_pad = pts_pad.to(device)
            s1p_pad = s1p_pad.to(device)
            lbl_pad = lbl_pad.to(device)
            scale_bt = scale_bt.to(device)
            B = pts_pad.shape[0]

            s2_logits = s2_model(pts_pad, s1p_pad, scale=scale_bt)

            for b in range(B):
                n_real = (lbl_pad[b] != -1).sum().item()
                s2_pred = s2_logits[b, :n_real].argmax(dim=-1)
                kl = lbl_pad[b, :n_real]
                s1_pred = s1p_pad[b, :n_real].argmax(dim=-1)

                combined = s2_pred.clone()
                s1_conc = (s1_pred == 1)
                combined[s1_conc] = 1

                all_cp.append(combined.cpu())
                all_ct.append(kl.cpu())

                non_conc = kl != 1
                if non_conc.any():
                    all_sp.append(s2_pred[non_conc].cpu())
                    all_st.append(kl[non_conc].cpu())

            prog.update(vtask, advance=1)

    if not all_cp:
        return _empty_results(class_names)

    cp = torch.cat(all_cp)
    ct = torch.cat(all_ct)
    c_ious = compute_iou(cp, ct, class_names)
    c_acc = compute_accuracy(cp, ct)
    c_prf = compute_precision_recall_f1(cp, ct, class_names)

    s_ious = _compute_s2_pure_ious(all_sp, all_st)

    rs_inter = ((cp == 2) & (ct == 2)).sum().float() + ((cp == 3) & (ct == 3)).sum().float()
    rs_union = ((cp == 2) | (ct == 2)).sum().float() + ((cp == 3) | (ct == 3)).sum().float()
    rs_iou = (rs_inter / (rs_union + 1e-10)).item()

    cm = cm_iou(c_ious.get("混凝土", 0), c_ious.get("钢筋", 0), c_ious.get("套筒", 0))

    return c_ious, c_prf, c_acc, s_ious, rs_iou, cm


def _compute_s2_pure_ious(all_sp, all_st):
    """计算 S2 纯 (非混凝土) IoU."""
    if not all_sp:
        s_ious = {"mIoU": 0.0}
        for n in ["噪声(S2)", "钢筋(S2)", "套筒(S2)"]:
            s_ious[n] = 0.0
        return s_ious

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
    return s_ious


def _empty_results(class_names):
    z = {k: 0.0 for k in list(class_names.values()) + ["mIoU"]}
    z2 = {k: {"precision": 0, "recall": 0, "f1": 0} for k in class_names.values()}
    return z, z2, 0.0, z, z2, 0.0, 0.0


def print_v2(ep, tl, c_ious, c_prf, c_acc, s_ious, rs_iou, cm, console=None):
    """打印 V2 验证结果."""
    if console is None:
        from rich.console import Console
        console = Console()
    cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
    cpr = " | ".join(
        f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}"
        for k, v in c_prf.items()
    )
    console.print(f"  S2: tl={tl:.4f} acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}] | cm={cm:.3f} | rsIoU={rs_iou:.3f}")
    console.print(f"    PRF(组合): [{cpr}]")
    sio = " | ".join(f"{k}={v:.3f}" for k, v in s_ious.items() if k != "mIoU")
    console.print(f"    S2纯: [{sio}] mIoU(S2)={s_ious.get('mIoU', 0):.3f}")
