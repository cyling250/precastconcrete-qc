# -*- coding: utf-8 -*-
"""
E5 阶段 2: 深度模型的全云逐点预测.

补片方式与评测协议一致 (球查询 8192 点, <8192 的余量丢弃),
返回 (逐点标签, 覆盖掩码) — 掩码外的点不参与测量, 保证与评测口径一致.
级联融合逻辑与 lib/cascade/validate.val_v2_simple 一致:
S2 输出 + S1 判混凝土的点直接置混凝土.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.utils import DEVICE, CHECKPOINT_BASE           # noqa: E402
from lib.cascade import Stage2Model                     # noqa: E402
from experiments._s1_trainer import make_s1_model       # noqa: E402
from experiments._a2_trainer import SingleStageModel    # noqa: E402

NUM_POINTS = 8192
VOXEL_MM = 40.0
PATCH_RADIUS_MM = 150.0


def _greedy_patches(pts: np.ndarray, rng: np.random.Generator):
    """复刻 RandomValDataset 的球查询贪心划分, 返回每个补片的全局索引列表."""
    n = len(pts)
    tree = cKDTree(pts)
    available = np.ones(n, dtype=bool)
    patches = []
    n_fail = 0
    while available.sum() >= NUM_POINTS and n_fail < 50:
        avail = np.where(available)[0]
        seed = rng.choice(avail)
        ball = tree.query_ball_point(pts[seed], r=PATCH_RADIUS_MM)
        cand = np.array([i for i in ball if available[i]], dtype=np.int64)
        if len(cand) >= NUM_POINTS:
            chosen = rng.choice(cand, NUM_POINTS, replace=False)
        else:
            need = min(NUM_POINTS * 2, len(avail))
            _, knn = tree.query(pts[seed].reshape(1, -1), k=need)
            chosen_set = set(cand.tolist())
            for i in knn[0]:
                if available[i]:
                    chosen_set.add(int(i))
                    if len(chosen_set) >= NUM_POINTS:
                        break
            chosen = np.array(list(chosen_set)[:NUM_POINTS], dtype=np.int64)
        if len(chosen) < NUM_POINTS:
            n_fail += 1
            continue
        n_fail = 0
        available[chosen] = False
        patches.append(np.sort(chosen))
    covered = np.zeros(n, dtype=bool)
    for p in patches:
        covered[p] = True
    return patches, covered


def _cascade_keep_mask(xyz: torch.Tensor, s1_prob: torch.Tensor, scale: float):
    """与 lib/cascade/filter.filter_patch 等价的保留掩码 (含索引)."""
    s1_pred = s1_prob.argmax(dim=-1)
    keep = torch.ones(len(s1_pred), dtype=torch.bool, device=xyz.device)
    conc_idx = torch.where(s1_pred == 1)[0]
    if conc_idx.numel():
        conc_pts = xyz[0][conc_idx]
        vn = VOXEL_MM / scale
        vi = torch.floor(conc_pts / vn).long()
        vi = vi - vi.min(dim=0).values
        mv = vi.max(dim=0).values + 1
        uid = vi[:, 0] * mv[1] * mv[2] + vi[:, 1] * mv[2] + vi[:, 2]
        su, si = torch.sort(uid)
        fi = torch.ones(len(su), dtype=torch.bool, device=xyz.device)
        fi[1:] = su[1:] != su[:-1]
        drop = torch.ones(len(conc_idx), dtype=torch.bool, device=xyz.device)
        drop[si[fi]] = False
        keep[conc_idx[drop]] = False
    return keep


@torch.no_grad()
def _predict_patches(patches, pts, s1_model=None, s2_model=None, m3_model=None):
    out = np.full(len(pts), -1, dtype=np.int64)
    for chosen in patches:
        p = pts[chosen]
        center = p.mean(axis=0)
        pn = p - center
        scale = float(np.max(np.linalg.norm(pn, axis=1))) + 1e-8
        pn = pn / scale
        xyz = torch.from_numpy(pn).float().unsqueeze(0).to(DEVICE)

        if m3_model is not None:
            pred = m3_model(xyz, scale=None)[0].argmax(-1)
        else:
            s1_logits = s1_model(xyz, scale=None)
            s1_prob = s1_logits.softmax(-1)[0]
            s1_pred = s1_prob.argmax(-1)
            keep = _cascade_keep_mask(xyz, s1_prob, scale)
            s2_logits = s2_model(xyz[:, keep], s1_prob[keep].unsqueeze(0),
                                 scale=torch.tensor([scale], device=DEVICE))
            s2_pred = s2_logits[0].argmax(-1)
            pred = torch.full((NUM_POINTS,), -1, dtype=torch.long, device=DEVICE)
            pred[keep] = s2_pred
            pred[s1_pred == 1] = 1
            valid = pred >= 0
            out[chosen[valid.cpu().numpy()]] = pred[valid].cpu().numpy()
            continue
        out[chosen] = pred.cpu().numpy()
    return out


def _load_state(model, path, key="model_state"):
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state.get(key, state) if isinstance(state, dict) else state)
    model.eval()
    return True


def load_deep_models():
    """加载 (s1, s2, m3); 缺权重时对应项为 None."""
    s1 = make_s1_model("pointmlp").to(DEVICE)
    ok1 = _load_state(s1, os.path.join(CHECKPOINT_BASE, "phase1_baseline_pointmlp", "best_s1.pth"))
    if not ok1:
        return (None, None, None)
    s2 = Stage2Model().to(DEVICE)
    ok2 = _load_state(s2, os.path.join(CHECKPOINT_BASE, "phase2_baseline_pointmlp", "best_s2.pth"))
    if not ok2:
        return (s1, None, None)
    m3 = None
    p3 = os.path.join(CHECKPOINT_BASE, "ablation_A2", "best.pth")
    if os.path.exists(p3):
        m3 = SingleStageModel(num_classes=4, in_channels=3).to(DEVICE)
        _load_state(m3, p3)
    return (s1, s2, m3)


def predict_cascade(pts: np.ndarray, s1_model, s2_model):
    """级联全云预测 → (逐点标签, 覆盖掩码)."""
    rng = np.random.default_rng(42)
    patches, covered = _greedy_patches(pts, rng)
    pred = _predict_patches(patches, pts, s1_model=s1_model, s2_model=s2_model)
    return pred, covered


def predict_m3(pts: np.ndarray, m3_model):
    """单阶段四类全云预测 → (逐点标签, 覆盖掩码)."""
    rng = np.random.default_rng(42)
    patches, covered = _greedy_patches(pts, rng)
    pred = _predict_patches(patches, pts, m3_model=m3_model)
    return pred, covered
