"""
上界诊断: 用真值套筒点直接拟合圆柱, 看可见套筒点是否构成闭合圆管.

若真值点本身的角度覆盖率就上不去, 说明该合成密度下套筒点不构成闭合管壁,
任何半径-RANSAC 都无法把套筒与混凝土分开 —— 这本身就是几何方法的适用边界.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from lib.pointcloud_generator._builder import BUILDERS, _filter_internal_rebar
from lib.pointcloud_generator._config import DENSITY_MAP
from lib.pointcloud_generator._noise import apply_density_variation

from lib.baselines.geometric_seg import (
    _cyl_radial,
    _fit_circle_2d,
    _plane_basis,
    local_pca_features,
)


def components(pts: np.ndarray, labels: np.ndarray, cls: int, radius: float) -> list[np.ndarray]:
    idx = np.flatnonzero(labels == cls)
    if len(idx) == 0:
        return []
    tree = cKDTree(pts[idx])
    seen = np.zeros(len(idx), dtype=bool)
    out = []
    for i in range(len(idx)):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in tree.query_ball_point(pts[idx][v], r=radius):
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        out.append(idx[np.array(comp)])
    return out


for dens in ["low", "medium", "high"]:
    rng = np.random.default_rng(42)
    res = BUILDERS["rectangular_column"](density=DENSITY_MAP[dens], rng=rng, variant="sleeve")
    pts, lbls = _filter_internal_rebar(res["points"], res["labels"])
    pts, lbls = apply_density_variation(pts, lbls, rng)
    feat = local_pca_features(pts, 24)

    print(f"\n===== 密度 {dens} =====  套筒点总数={int((lbls == 3).sum())}  "
          f"混凝土={int((lbls == 1).sum())}  随机噪点={int(np.mean(lbls == 1) * 0)}")
    comps = components(pts, lbls, 3, radius=20.0)
    sizes = sorted((len(c) for c in comps), reverse=True)
    print(f"套筒点连通簇数(20mm)={len(comps)}, 最大几个簇={sizes[:12]}")

    print(f"{'簇#':<5}{'点数':>7}{'拟合半径mm':>12}{'覆盖率°':>10}{'轴向长mm':>10}")
    print("-" * 46)
    for ci, comp in enumerate(sorted(comps, key=len, reverse=True)[:10]):
        p = pts[comp]
        n = feat["normal"][comp]
        nn = n - n.mean(axis=0)
        w, v = np.linalg.eigh(nn.T @ nn)
        d = v[:, 0]
        u, vv = _plane_basis(d)
        c0 = p.mean(axis=0)
        rel = p - c0
        fit = _fit_circle_2d(np.column_stack([rel @ u, rel @ vv]))
        if fit is None:
            print(f"{ci:<5}{len(comp):>7}   拟合失败")
            continue
        c2, r = fit
        a = c0 + c2[0] * u + c2[1] * vv
        perp = rel - np.outer(rel @ d, d)
        ang = np.arctan2(perp @ vv, perp @ u)
        cov = 15.0 * np.unique(np.floor((ang + np.pi) / np.deg2rad(15.0)).astype(np.int64)).size
        span = float(np.ptp((p - a) @ d))
        res_r = _cyl_radial(p, a, d)
        print(f"{ci:<5}{len(comp):>7}{r:>12.1f}{cov:>10.0f}{span:>10.1f}   "
              f"径向残差RMS={np.sqrt(np.mean((res_r - r) ** 2)):.1f}mm")
