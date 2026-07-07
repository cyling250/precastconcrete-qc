"""
级联管线体素降采样: S1 预测 → 仅混凝土降采样, 保留全部噪声+金属.
"""

import torch


def filter_patch(pts_norm: torch.Tensor, s1_probs: torch.Tensor,
                 gt_labels: torch.Tensor, scale_mm: float,
                 voxel_mm: float = 40.0):
    """S1 预测 → 混凝土 voxel_mm 降采样 + 保留全部噪声/金属.

    Args:
        pts_norm: (N, 3) normalized points
        s1_probs: (N, 3) S1 softmax probabilities
        gt_labels: (N,) ground truth labels
        scale_mm: float, normalization scale in mm
        voxel_mm: float, voxel size in mm (default 40mm = 4cm)
    Returns:
        filtered_pts, filtered_probs, filtered_labels
    """
    N = len(gt_labels)
    s1_pred = s1_probs.argmax(dim=-1)  # (N,) {0=噪, 1=砼, 2=金}

    keep = torch.ones(N, dtype=torch.bool, device=pts_norm.device)

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
        drop = torch.ones(len(conc_idx), dtype=torch.bool, device=pts_norm.device)
        drop[si[fi]] = False
        keep[conc_idx[drop]] = False

    return pts_norm[keep], s1_probs[keep], gt_labels[keep]
