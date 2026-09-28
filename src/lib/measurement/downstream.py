# -*- coding: utf-8 -*-
"""
下游尺寸测量管线 (E5): 从分割结果到钢筋/箍筋间距与套筒定位.

只依赖点云几何, 不依赖网络. 三个测量任务:
  1) 箍筋间距: 金属点沿构件轴线的密度投影峰值间隔 (排除两端纵筋外露段);
  2) 纵筋间距: 端部截面内金属点聚类中心的两两最近邻距离中位数;
  3) 套筒定位: 预测套筒点簇中心与真值簇中心的匹配偏移.

几何约定 (与 lib/pointcloud_generator 一致):
  柱: 轴向 = Z;  梁: 轴向 = X (运行时按包围盒最长维自动判定).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# ── 容差 (mm). 来源: GB 50204-2015 混凝土结构工程施工质量验收规范
# 钢筋安装允许偏差与预埋件允许偏差条款.
# ⚠️ TO VERIFY: 投稿前必须由作者对照标准原文核对数值与条款号! ──
TOLERANCE_MM = {
    "stirrup_spacing": 10.0,   # 箍筋间距允许偏差 (习惯引用值 ±10)
    "rebar_spacing": 10.0,     # 受力钢筋间距允许偏差 (习惯引用值 ±10)
    "sleeve_position": 5.0,    # 预埋套筒/预埋件中心线位置 (习惯引用值 5)
}


def _dominant_axis(pts: np.ndarray) -> int:
    return int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))


def connected_clusters(pts: np.ndarray, radius: float = 25.0) -> list[np.ndarray]:
    """按距离阈值做连通分量聚类, 返回每簇的点索引列表 (按簇大小降序)."""
    n = len(pts)
    if n == 0:
        return []
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = [np.array(v) for v in groups.values()]
    out.sort(key=len, reverse=True)
    return out


def measure_stirrup_spacing(metal_pts: np.ndarray,
                            bin_mm: float = 2.0,
                            min_sep_mm: float = 40.0,
                            trim_frac: float = 0.15) -> dict:
    """箍筋间距: 轴向密度投影峰值间隔 (中位数).

    排除两端 trim_frac 比例的长度 (纵筋外露段集中于此).
    返回 {spacing_mm, n_stations}; 无有效峰值时 spacing_mm=nan.
    """
    if len(metal_pts) < 100:
        return {"spacing_mm": float("nan"), "n_stations": 0}
    ax = _dominant_axis(metal_pts)
    t = metal_pts[:, ax]
    L = t.max() - t.min()
    lo, hi = t.min() + trim_frac * L, t.max() - trim_frac * L
    t_mid = t[(t >= lo) & (t <= hi)]
    if len(t_mid) < 100:
        return {"spacing_mm": float("nan"), "n_stations": 0}

    edges = np.arange(t_mid.min(), t_mid.max() + bin_mm, bin_mm)
    hist, _ = np.histogram(t_mid, bins=edges)
    k = 5
    kernel = np.ones(k) / k
    smooth = np.convolve(hist, kernel, mode="same")
    thresh = 0.3 * smooth.max()

    centers = (edges[:-1] + edges[1:]) / 2
    # 1) 所有局部极大 (高于阈值); 2) 相距 < min_sep 的合并为较高的那个
    cand = [centers[i] for i in range(1, len(smooth) - 1)
            if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
            and smooth[i] > thresh]
    cand_val = {c: smooth[np.searchsorted(centers, c)] for c in cand}
    peaks: list[float] = []
    for c in cand:
        if peaks and c - peaks[-1] < min_sep_mm:
            if cand_val[c] > cand_val[peaks[-1]]:
                peaks[-1] = c
        else:
            peaks.append(c)
    if len(peaks) < 2:
        return {"spacing_mm": float("nan"), "n_stations": len(peaks)}
    diffs = np.diff(sorted(peaks))
    return {"spacing_mm": float(np.median(diffs)), "n_stations": len(peaks)}


def measure_longitudinal_spacing(metal_pts: np.ndarray,
                                 end_frac: float = 0.08,
                                 cluster_radius: float = 30.0,
                                 min_pts_per_bar: int = 30,
                                 min_axial_extent: float = 80.0,
                                 max_inplane_extent: float = 100.0) -> dict:
    """纵筋间距: 端部外露钢筋簇中心的最近邻距离中位数 (mm).

    与 3D 聚类 (非截面投影) 抗污染: 只保留"棒状"簇 (轴向跨度 ≥ min_axial_extent
    且截面内尺寸 ≤ max_inplane_extent), 排除箍筋环 / 噪声团等非纵筋簇.

    返回 {spacing_mm, n_bars}; 检出 <2 根时 spacing_mm=nan.
    """
    if len(metal_pts) < 100:
        return {"spacing_mm": float("nan"), "n_bars": 0}
    ax = _dominant_axis(metal_pts)
    t = metal_pts[:, ax]
    L = t.max() - t.min()
    inplane_axes = [i for i in range(3) if i != ax]
    for sel in (t >= t.max() - end_frac * L, t <= t.min() + end_frac * L):
        sec = metal_pts[sel]
        if len(sec) < min_pts_per_bar:
            continue
        bars = []
        for c in connected_clusters(sec, cluster_radius):
            if len(c) < min_pts_per_bar:
                continue
            span = sec[c].max(axis=0) - sec[c].min(axis=0)
            if span[ax] >= min_axial_extent and max(span[inplane_axes]) <= max_inplane_extent:
                bars.append(sec[c][:, inplane_axes].mean(axis=0))
        if len(bars) >= 2:
            centers = np.array(bars)
            tree = cKDTree(centers)
            d, _ = tree.query(centers, k=2)
            return {"spacing_mm": float(np.median(d[:, 1])), "n_bars": len(bars)}
    return {"spacing_mm": float("nan"), "n_bars": 0}


def measure_hairpin_spacing(metal_pts: np.ndarray,
                            bin_mm: float = 2.0,
                            min_sep_mm: float = 40.0,
                            trim_frac: float = 0.08) -> dict:
    """叠合梁外露 U 形箍筋(发夹筋)沿梁轴的间距.

    生成器设计为非均匀间距: 端部 0.6×s、中部 1.4×s、0.25L 过渡,
    故分端部区(两端各 0.25L)与中部区(中间 0.5L)分别报告中位数间距.

    返回 {end_mm, mid_mm, n_stations}; 某区峰值不足时为 nan.
    """
    if len(metal_pts) < 100:
        return {"end_mm": float("nan"), "mid_mm": float("nan"), "n_stations": 0}
    ax = _dominant_axis(metal_pts)
    t = metal_pts[:, ax]
    t0, t1 = t.min(), t.max()
    L = t1 - t0

    def peaks_in(lo: float, hi: float) -> list:
        tt = t[(t >= lo) & (t <= hi)]
        if len(tt) < 30:
            return []
        edges = np.arange(tt.min(), tt.max() + bin_mm, bin_mm)
        if len(edges) < 4:
            return []
        hist, _ = np.histogram(tt, bins=edges)
        k = 5
        smooth = np.convolve(hist, np.ones(k) / k, mode="same")
        thresh = 0.3 * smooth.max()
        centers = (edges[:-1] + edges[1:]) / 2
        cand = [centers[i] for i in range(1, len(smooth) - 1)
                if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
                and smooth[i] > thresh]
        cand_val = {c: smooth[np.searchsorted(centers, c)] for c in cand}
        pk: list = []
        for c in cand:
            if pk and c - pk[-1] < min_sep_mm:
                if cand_val[c] > cand_val[pk[-1]]:
                    pk[-1] = c
            else:
                pk.append(c)
        return pk

    end_lo, end_hi = t0 + trim_frac * L, t0 + 0.25 * L
    end_hi2, end_lo2 = t1 - 0.25 * L, t1 - trim_frac * L
    mid_lo, mid_hi = t0 + 0.375 * L, t1 - 0.375 * L

    end_pk = peaks_in(end_lo, end_hi) + peaks_in(end_hi2, end_lo2)
    mid_pk = peaks_in(mid_lo, mid_hi)

    def med(pk):
        return float(np.median(np.diff(sorted(pk)))) if len(pk) >= 2 else float("nan")

    return {"end_mm": med(end_pk), "mid_mm": med(mid_pk),
            "n_stations": len(end_pk) + len(mid_pk)}


def measure_sleeve_offsets(pred_pts: np.ndarray, gt_pts: np.ndarray,
                           cluster_radius: float = 40.0,
                           min_pts: int = 30) -> dict:
    """套筒定位: 预测簇中心 ↔ 最近真值簇中心的偏移 (mm).

    返回 {offsets_mm[], n_gt, n_pred_matched, n_gt_matched, n_pred}.
    """
    gt_cl = [c for c in connected_clusters(gt_pts, cluster_radius) if len(c) >= min_pts] \
        if len(gt_pts) else []
    pr_cl = [c for c in connected_clusters(pred_pts, cluster_radius) if len(c) >= min_pts] \
        if len(pred_pts) else []
    if not gt_cl:
        return {"offsets_mm": [], "n_gt": 0, "n_pred": len(pr_cl),
                "n_gt_matched": 0, "n_pred_matched": 0}
    gt_c = np.array([gt_pts[c].mean(axis=0) for c in gt_cl])
    offsets, matched_pred = [], set()
    for c in gt_c:
        if pr_cl:
            pr_c = np.array([pred_pts[p].mean(axis=0) for p in pr_cl])
            d = np.linalg.norm(pr_c - c, axis=1)
            j = int(np.argmin(d))
            offsets.append(float(d[j]))
            matched_pred.add(j)
        else:
            offsets.append(float("nan"))
    return {"offsets_mm": offsets, "n_gt": len(gt_cl), "n_pred": len(pr_cl),
            "n_gt_matched": sum(not np.isnan(o) for o in offsets),
            "n_pred_matched": len(matched_pred)}
