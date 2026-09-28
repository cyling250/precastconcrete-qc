"""
几何基线: PCA 局部几何特征 + RANSAC 圆柱/平面拟合的材料级点云分割.

目标类别与主模型一致: {0 噪声, 1 混凝土, 2 钢筋, 3 套筒}.

用途
----
1. 作为 PC-CascadeNet 的经典方法对照 (审稿人必问: 深度学习相对精心调参的
   几何方法好在哪、值不值);
2. 支撑"噪声-退化交叉点"分析: 几何方法在低噪声下可用, 高噪声下失效,
   交叉点即深度学习的价值区间.

方法路线沿用本文引言引用的几何方法谱系:
  - PCA 特征值描述子 (linearity / planarity / sphericity) 分离柱状钢筋与平面混凝土
    (Kim et al. 2020, Autom. Constr. 112:103077; Wang et al. 2017, CACAIE 32:787)
  - RANSAC 圆柱拟合 [Fischler & Bolles 1981] 提取钢筋/套筒轴线并按半径分流
  - 统计离群点剔除 (SOR) 识别扫描噪点

性能说明: 本数据集的点间距约 7 mm (0.02 pts/mm^2), 已与最小钢筋直径同量级,
体素降采样无法在保形的前提下减少点数. 因此模型提议与内点计数在一个随机
子采样"搜索集"上完成, 最终内点在完整工作点云上重算, 以把贪心 RANSAC 的
外层循环代价压到可接受范围.

依赖: numpy + scipy, 纯 CPU, 不需要 torch / CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# 与生成器保持一致的标签体系
LABEL_NOISE = 0
LABEL_CONCRETE = 1
LABEL_REBAR = 2
LABEL_SLEEVE = 3


# ═══════════════════════════════════════════════════════════
#  参数
# ═══════════════════════════════════════════════════════════


@dataclass
class GeometricParams:
    """几何基线超参数.

    默认值按数据集已知量级设定 (单位 mm):
      - 纵筋半径 8~16 mm, 箍筋半径 4~6 mm, 套筒外半径 24~33 mm, 内腔半径 12.5~27 mm
      - 混凝土保护层 25 mm, 露筋长度 200 mm, 套筒长度 310~570 mm
      - 混凝土表面粗糙度 sigma=0.4 mm, 扫描抖动 sigma<=10 mm

    审稿口径: 这些参数应在验证集上扫描后报最优值, 而不是直接采用默认值.
    """

    # ── 预处理 ──
    voxel_mm: float = 4.0          # 体素降采样 (仅在点数超过 max_points 时生效)
    max_points: int = 200_000
    knn: int = 24                  # 局部 PCA / 法向量估计邻域点数
    feat_radius_mm: float = 40.0   # 固定物理邻域半径 (跨密度复用阈值的关键)
    feature_chunk: int = 60_000

    # ── 噪声识别 (SOR) ──
    sor_k: int = 12
    sor_alpha: float = 3.5         # d_mean > mu + alpha * 1.4826*MAD 判为噪声

    # ── 判别阈值 (由 diag_features 在本数据上实测标定) ──
    # 钢筋: 线性度 P10=0.71 / 中位 0.93; 混凝土在 lin>=0.80 处误纳率仅 0.04%.
    # 套筒: 半径大 (24~33 mm) 导致局部近平面, 线性度中位仅 0.37, 与混凝土 (0.34) 不可分;
    #       但曲率显著抬高 (中位 0.07~0.12 vs 混凝土 0.0005), 故用曲率作套筒候选.
    lin_min: float = 0.80          # 钢筋: 线性度阈值 (经典 PCA 线性度判别)
    sleeve_curv_min: float = 0.01  # 套筒: 曲率候选阈值 (召回 ~88-91%, 误纳 6-8%)

    # ── 三分类 (噪声/混凝土/金属) 模式 ──
    # 实测结论: 套筒的可见点集包含端面圆盘与内壁, 壁厚仅 6 mm 而对抗扫描抖动所需
    # 的径向容差同量级, 单一圆柱模型无法把内/外壁分开; 且混凝土点数以约 20:1 压制
    # 套筒, 贪心 RANSAC 恒选中混凝土长边. 因此四分类中"套筒"一类几何方法给不出,
    # 只能退化为 Stage 1 同口径的三分类 (金属 = 钢筋 ∪ 套筒).
    three_class: bool = True
    metal_curv_min: float = 0.02   # 金属 (含套筒) 的补充判据: 曲率阈值

    # ── 圆柱 RANSAC (仅用于套筒; 钢筋点数过碎, 不足以支撑稳健拟合) ──
    # 实测: low 密度下钢筋点被切成 89 个簇, 中位仅 12 点, 无法做圆柱 RANSAC.
    #       套筒单管在 medium 密度下约 900~1500 点, 支撑充足.
    cyl_search_max: int = 25_000   # 搜索集上限 (随机子采样)
    cyl_iters: int = 3000          # 模型提议次数
    cyl_tol_mm: float = 2.5        # 径向内点容差
    cyl_min_support: int = 150     # 全量点云上的最小内点数
    cyl_max_models: int = 40
    cyl_topk: int = 40             # 每次按内点数降序检验的提议个数
    cyl_batch: int = 64
    cyl_angle_min_deg: float = 20.0
    rebar_radius_range: tuple[float, float] = (3.0, 18.0)
    sleeve_radius_range: tuple[float, float] = (19.0, 36.0)
    sleeve_fill_interior: bool = True   # 套筒: 轴向范围内径向 <= r 的点一并归入
    # 套筒是闭合圆管: 内点应环绕轴线几乎一整圈. 混凝土面被误拟合时只是轴向相切的
    # 局部弧段, 覆盖角很小 —— 用角度覆盖率把二者分开, 这是最有效的单一判别.
    sleeve_min_coverage_deg: float = 240.0
    sleeve_max_len_mm: float = 800.0    # 套筒长度 310~570 mm, 超出即判为误拟合

    # ── 平面 RANSAC (混凝土面) ──
    plane_search_max: int = 25_000
    plane_iters: int = 1500
    plane_tol_mm: float = 3.0
    plane_adaptive_tol: bool = True     # 按拟合残差 MAD 自适应放宽容差
    plane_tol_max_mm: float = 25.0
    plane_min_support: int = 3_000
    plane_max_models: int = 16
    plane_batch: int = 96

    # ── 兜底分配 ──
    vote_k: int = 12

    seed: int = 42


# ═══════════════════════════════════════════════════════════
#  基础工具
# ═══════════════════════════════════════════════════════════


def voxel_downsample(points: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """体素降采样.

    Returns:
        (降采样后的点, 反向映射). 反向映射 inv 长度为原始点数, inv[i] 给出原始第 i 个
        点所属的降采样点下标, 便于把降采样云上的标签传回全量点云.
    """
    if voxel <= 0 or len(points) == 0:
        return points, np.arange(len(points))
    keys = np.floor(points / voxel).astype(np.int64)
    # 注意: numpy 的 unique(axis=0, return_inverse=True) 返回的 inv 在部分版本里是
    # (N,1) 形状, 必须 ravel.
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    rep = np.full(len(uniq), 0, dtype=np.int64)
    order = np.argsort(inv, kind="stable")
    first = np.concatenate([[True], np.diff(inv[order]) != 0])
    rep[inv[order][first]] = order[first]
    return points[rep], inv


def local_pca_features(
    points: np.ndarray,
    knn: int,
    chunk: int = 60_000,
    radius_mm: float | None = None,
    kmax: int = 64,
) -> dict[str, np.ndarray]:
    """邻域协方差的主成分特征.

    Args:
        knn: 固定邻域点数 (radius_mm 为 None 时生效).
        radius_mm: 固定邻域半径 (mm). 给定时改为"半径内最多 kmax 个邻居"的邻域,
            并取至少 6 个邻居计算 PCA.

    为什么要支持固定半径: kNN 邻域会随点云密度变化而改变物理尺度, 使 PCA 描述子的
    判别能力依赖扫描分辨率 —— 实测中同一套阈值在 low/medium/high 三档密度下的
    金属类 IoU 从 0.60 掉到 0.12. 这正是引言所述"几何方法需要针对不同扫描分辨率
    重新调参"的根本原因. 固定物理半径可以让阈值跨密度复用, 也是对几何方法更公平的
    设定 (工程实践中同样按物理尺度选邻域).

    返回 dict:
      normal      (N,3) 法向量 (最小特征值方向)
      linearity   (N,)  (l1-l2)/l1
      planarity   (N,)  (l2-l3)/l1
      sphericity  (N,)  l3/l1
      curvature   (N,)  l3/(l1+l2+l3)
      knn_dist    (N,)  到 k 近邻的平均距离 (供 SOR 使用)
      n_nb        (N,)  实际参与 PCA 的邻居数
    """
    tree = cKDTree(points)
    n = len(points)
    use_radius = radius_mm is not None and radius_mm > 0
    k = min(kmax if use_radius else knn + 1, n)
    k = max(k, 2)

    normal = np.zeros((n, 3))
    lin = np.zeros(n)
    pla = np.zeros(n)
    sph = np.zeros(n)
    cur = np.zeros(n)
    kd = np.zeros(n)
    n_nb = np.zeros(n)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dists, idx = tree.query(points[s:e], k=k, workers=-1)
        if k == 2:
            kd[s:e] = np.asarray(dists)[:, -1]
            continue
        kd[s:e] = dists[:, 1:min(knn + 1, k)].mean(axis=1)
        nb = points[idx]                                   # (B,k,3)
        b = e - s
        if use_radius:
            w = (dists <= radius_mm).astype(np.float64)
            w[:, 0] = 0.0                                  # 去掉自身
            cnt = w.sum(axis=1)
            # 邻居太少的点没有有效的局部几何, 退回全 k 邻域
            few = cnt < 6
            if few.any():
                w[few] = 1.0
                w[few, 0] = 0.0
                cnt[few] = k - 1
            n_nb[s:e] = cnt
            mean = np.einsum("bk,bki->bi", w, nb) / cnt[:, None]
            dev = (nb - mean[:, None, :]) * w[:, :, None]
            cov = np.einsum("bki,bkj->bij", dev, dev) / np.maximum(cnt - 1, 1)[:, None, None]
        else:
            n_nb[s:e] = k - 1
            nb = nb - nb.mean(axis=1, keepdims=True)
            cov = np.einsum("bki,bkj->bij", nb, nb) / max(k - 1, 1)

        wv, v = np.linalg.eigh(cov)                        # 特征值升序
        l1, l2, l3 = wv[:, 2], wv[:, 1], wv[:, 0]
        eps = 1e-12
        lin[s:e] = (l1 - l2) / (l1 + eps)
        pla[s:e] = (l2 - l3) / (l1 + eps)
        sph[s:e] = l3 / (l1 + eps)
        cur[s:e] = l3 / (l1 + l2 + l3 + eps)
        normal[s:e] = v[:, :, 0]

    return {
        "normal": normal,
        "linearity": lin,
        "planarity": pla,
        "sphericity": sph,
        "curvature": cur,
        "knn_dist": kd,
        "n_nb": n_nb,
    }


def statistical_outlier_mask(knn_dist: np.ndarray, alpha: float) -> np.ndarray:
    """SOR: 平均近邻距离显著偏大的点判为噪声 (尺度用 MAD 稳健估计)."""
    mu = np.median(knn_dist)
    mad = np.median(np.abs(knn_dist - mu))
    sigma = 1.4826 * mad
    if sigma < 1e-9:
        sigma = float(knn_dist.std()) + 1e-9
    return knn_dist > mu + alpha * sigma


def _fit_circle_2d(xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Kasa 代数圆拟合, 返回 (圆心(2,), 半径)."""
    if len(xy) < 3:
        return None
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = -sol[0] / 2.0, -sol[1] / 2.0
    val = cx * cx + cy * cy - sol[2]
    if val <= 0:
        return None
    return np.array([cx, cy]), float(np.sqrt(val))


def _plane_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """给定单位向量 d, 返回与 d 正交的两个单位基向量."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(d[0]) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = a - np.dot(a, d) * d
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(d, u)
    return u, v


def _subsample_search(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """返回用于模型提议的索引子集 (按点数上限随机子采样)."""
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))


def _cyl_radial(points: np.ndarray, a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """点到轴线的径向距离."""
    rel = points - a
    axial = rel @ d
    return np.linalg.norm(rel - np.outer(axial, d), axis=1)


def _cyl_axial(points: np.ndarray, a: np.ndarray, d: np.ndarray) -> np.ndarray:
    return (points - a) @ d


# ═══════════════════════════════════════════════════════════
#  圆柱 RANSAC
# ═══════════════════════════════════════════════════════════


def _propose_cylinders(
    sp: np.ndarray, sn: np.ndarray, rng: np.random.Generator, p: GeometricParams
) -> np.ndarray:
    """批量提议圆柱模型, 返回 (M,7) = [a(3), d(3), r].

    两点法向的叉积给出轴向; 由 p1 - r*n1 = p2 - r*n2 解出半径.

    PCA 法向量的符号是任意的 (套筒内壁朝内、外壁朝外), 因此同时输出
    a = p1 - r*n1 与镜像解 a = p1 + r*n1, 由内点计数挑选正确的一侧.
    """
    m = len(sp)
    if m < 2:
        return np.empty((0, 7))
    i = rng.integers(0, m, size=p.cyl_iters)
    j = rng.integers(0, m, size=p.cyl_iters)
    ok = i != j
    i, j = i[ok], j[ok]
    if len(i) == 0:
        return np.empty((0, 7))
    p1, p2 = sp[i], sp[j]
    n1, n2 = sn[i], sn[j]

    cr = np.cross(n1, n2)
    cn = np.linalg.norm(cr, axis=1)
    valid = cn > np.sin(np.deg2rad(p.cyl_angle_min_deg))
    if not valid.any():
        return np.empty((0, 7))
    p1, p2 = p1[valid], p2[valid]
    n1, n2 = n1[valid], n2[valid]
    d = cr[valid] / cn[valid, None]

    rel = p1 - p2
    dn = n1 - n2
    rel_p = rel - np.sum(rel * d, axis=1)[:, None] * d
    dn_p = dn - np.sum(dn * d, axis=1)[:, None] * d
    num = np.linalg.norm(rel_p, axis=1)
    den = np.linalg.norm(dn_p, axis=1)
    good = den > 1e-6
    if not good.any():
        return np.empty((0, 7))
    r = num[good] / den[good]
    keep = (r >= p.rebar_radius_range[0]) & (r <= p.sleeve_radius_range[1])
    if not keep.any():
        return np.empty((0, 7))
    p1k, n1k, dk, rk = p1[good][keep], n1[good][keep], d[good][keep], r[keep]
    a_plus = p1k - rk[:, None] * n1k
    a_minus = p1k + rk[:, None] * n1k
    return np.vstack(
        [
            np.column_stack([a_plus, dk, rk]),
            np.column_stack([a_minus, dk, rk]),
        ]
    )


def _batch_cyl_counts(
    points: np.ndarray, models: np.ndarray, tol: float, batch: int
) -> np.ndarray:
    """批量统计圆柱薄壳内点数."""
    counts = np.zeros(len(models), dtype=np.int64)
    for s in range(0, len(models), batch):
        e = min(s + batch, len(models))
        mb = models[s:e]
        a, d, r = mb[:, :3], mb[:, 3:6], mb[:, 6]
        rel = points[None, :, :] - a[:, None, :]
        axial = np.einsum("bnk,bk->bn", rel, d)
        perp = rel - axial[:, :, None] * d[:, None, :]
        rad = np.linalg.norm(perp, axis=2)
        counts[s:e] = (np.abs(rad - r[:, None]) < tol).sum(axis=1)
    return counts


def _refine_cylinder(
    points: np.ndarray,
    normals: np.ndarray,
    inl: np.ndarray,
    d0: np.ndarray,
    tol: float,
    iters: int = 3,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """内点最小二乘细化: 法向协方差给轴向, 投影到正交平面后圆拟合给轴位与半径."""
    if inl.sum() < 3:
        return None
    d = d0
    for _ in range(iters):
        idx = np.flatnonzero(inl)
        if len(idx) < 3:
            return None
        nn = normals[idx]
        nn = nn - nn.mean(axis=0)
        w, v = np.linalg.eigh(nn.T @ nn)
        d_new = v[:, 0]
        if np.dot(d_new, d) < 0:
            d_new = -d_new
        d = d_new
        u, vv = _plane_basis(d)
        c0 = points[idx].mean(axis=0)
        rel = points[idx] - c0
        fit = _fit_circle_2d(np.column_stack([rel @ u, rel @ vv]))
        if fit is None:
            return None
        c2, r = fit
        a = c0 + c2[0] * u + c2[1] * vv
        inl = np.abs(_cyl_radial(points, a, d) - r) < tol
    return a, d, float(r)


def _tube_metrics(
    points: np.ndarray, a: np.ndarray, d: np.ndarray, inl: np.ndarray
) -> tuple[float, float]:
    """圆管的几何一致性指标: (角度覆盖率°, 轴向长度 mm)."""
    shell = np.flatnonzero(inl)
    if len(shell) == 0:
        return 0.0, 0.0
    u, vv = _plane_basis(d)
    rel = points[shell] - a
    perp = rel - np.outer(rel @ d, d)
    ang = np.arctan2(perp @ vv, perp @ u)
    bin_idx = np.floor((ang + np.pi) / np.deg2rad(15.0)).astype(np.int64)
    cov_deg = 15.0 * np.unique(bin_idx).size
    ax = (points[shell] - a) @ d
    return float(cov_deg), float(ax.max() - ax.min())


def ransac_cylinders(
    points: np.ndarray,
    normals: np.ndarray,
    eligible: np.ndarray,
    params: GeometricParams,
    rng: np.random.Generator,
    expect_radius: tuple[float, float] | None = None,
) -> list[dict]:
    """贪心 RANSAC 提取多个圆柱.

    Args:
        expect_radius: 接受模型必须落在此半径区间内. 缺少该约束时, 最小二乘细化会
            漂移到上百毫米的"伪圆柱"(实际是接近平面的混凝土面), 必须显式约束.

    关键实现选择: 候选集里混凝土边缘结构的点数往往多于真实的细管, 因此不能直接取
    "内点数最多"的提议就用, 否则永远选中混凝土长边. 这里改为按内点数降序逐个检验,
    只接受通过"半径区间 + 圆管角度覆盖率 + 轴向长度"三项几何校验的模型.
    全部内点计数与最小二乘细化都只在候选集上进行, 避免全量点云中混凝土把模型带偏.
    """
    models: list[dict] = []
    pool = eligible.copy()
    if expect_radius is None:
        r_lo, r_hi = params.rebar_radius_range[0], params.sleeve_radius_range[1]
    else:
        r_lo, r_hi = expect_radius

    for _ in range(params.cyl_max_models):
        cand = np.flatnonzero(pool)
        if len(cand) < 50:
            break
        sel = cand[_subsample_search(len(cand), params.cyl_search_max, rng)]
        frac = len(sel) / len(points)
        min_sup = max(30, int(params.cyl_min_support * frac))

        cp, cn = points[sel], normals[sel]
        prop = _propose_cylinders(cp, cn, rng, params)
        if len(prop) == 0:
            break
        counts = _batch_cyl_counts(cp, prop, params.cyl_tol_mm, params.cyl_batch)

        chosen = None
        for k in np.argsort(counts)[::-1][: params.cyl_topk]:
            if counts[k] < min_sup:
                break                      # 已按内点数降序, 后面只会更少
            m = prop[k]
            a0, d0, r0 = m[:3], m[3:6], float(m[6])
            seed = np.abs(_cyl_radial(cp, a0, d0) - r0) < params.cyl_tol_mm * 2.0
            ref = _refine_cylinder(cp, cn, seed, d0, params.cyl_tol_mm * 2.0)
            if ref is None:
                continue
            a, d, r = ref
            if not (r_lo <= r <= r_hi):
                continue
            inl_c = np.abs(_cyl_radial(cp, a, d) - r) < params.cyl_tol_mm
            if inl_c.sum() < min_sup:
                continue
            cov, span = _tube_metrics(cp, a, d, inl_c)
            if cov < params.sleeve_min_coverage_deg or span > params.sleeve_max_len_mm:
                continue
            chosen = (a, d, r, inl_c, cov, span)
            break

        if chosen is None:
            break

        a, d, r, inl_c, cov, span = chosen
        # 在候选集内扩到全部内点, 再映射回全量点云
        ax = _cyl_axial(points, a, d)
        rad = _cyl_radial(points, a, d)
        inl = (np.abs(rad - r) < params.cyl_tol_mm) & eligible
        if params.sleeve_fill_interior and r >= params.sleeve_radius_range[0]:
            t_lo, t_hi = np.percentile(ax[inl], [1, 99])
            seg = (ax >= t_lo - params.cyl_tol_mm) & (ax <= t_hi + params.cyl_tol_mm)
            inl = inl | (seg & (rad <= r + params.cyl_tol_mm))

        models.append(
            {
                "axis_point": a,
                "axis_dir": d,
                "radius": r,
                "inliers": inl,
                "coverage_deg": cov,
                "length_mm": span,
            }
        )
        pool[inl] = False

    return models


# ═══════════════════════════════════════════════════════════
#  平面 RANSAC
# ═══════════════════════════════════════════════════════════


def _refine_plane(points: np.ndarray, inl: np.ndarray) -> tuple[np.ndarray, float] | None:
    idx = np.flatnonzero(inl)
    if len(idx) < 3:
        return None
    p = points[idx]
    c = p.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((p - c).T))
    n = v[:, 0]
    return n, float(np.dot(n, c))


def ransac_planes(
    points: np.ndarray, eligible: np.ndarray, params: GeometricParams,
    rng: np.random.Generator,
) -> list[dict]:
    """贪心 RANSAC 提取多个平面 (混凝土表面), 容差按残差 MAD 自适应."""
    models: list[dict] = []
    pool = eligible.copy()

    for _ in range(params.plane_max_models):
        cand = np.flatnonzero(pool)
        if len(cand) < 50:
            break
        sel = cand[_subsample_search(len(cand), params.plane_search_max, rng)]
        frac = len(sel) / len(points)
        min_sup = max(50, int(params.plane_min_support * frac))
        p0 = points[sel]

        m = len(sel)
        i = rng.integers(0, m, size=params.plane_iters)
        j = rng.integers(0, m, size=params.plane_iters)
        k = rng.integers(0, m, size=params.plane_iters)
        ok = (i != j) & (j != k) & (i != k)
        i, j, k = i[ok], j[ok], k[ok]
        if len(i) == 0:
            break
        nrm = np.cross(p0[j] - p0[i], p0[k] - p0[i])
        nn = np.linalg.norm(nrm, axis=1)
        good = nn > 1e-9
        if not good.any():
            break
        nnrm = nrm[good] / nn[good, None]
        off = np.einsum("ij,ij->i", nnrm, p0[i[good]])

        counts = np.zeros(len(nnrm), dtype=np.int64)
        for s in range(0, len(nnrm), params.plane_batch):
            e = min(s + params.plane_batch, len(nnrm))
            dist = np.abs(p0 @ nnrm[s:e].T - off[s:e][None, :])
            counts[s:e] = (dist < params.plane_tol_mm).sum(axis=0)
        best = int(np.argmax(counts))
        if counts[best] < min_sup:
            break

        n_b, o_b = nnrm[best], float(off[best])
        inl = np.abs(points @ n_b - o_b) < params.plane_tol_mm
        ref = _refine_plane(points, inl)
        if ref is None:
            pool[sel] = False
            continue
        n_b, o_b = ref
        tol = params.plane_tol_mm
        if params.plane_adaptive_tol:
            res = points @ n_b - o_b
            near = res[np.abs(res) < 3 * params.plane_tol_mm]
            mad = np.median(np.abs(near)) if len(near) else 0.0
            sigma = 1.4826 * mad if mad > 1e-9 else 0.0
            tol = float(np.clip(3.0 * sigma, params.plane_tol_mm, params.plane_tol_max_mm))
        inl = np.abs(points @ n_b - o_b) < tol
        if inl.sum() < params.plane_min_support:
            break
        models.append({"normal": n_b, "offset": o_b, "tol": tol, "inliers": inl})
        pool[inl] = False

    return models


# ═══════════════════════════════════════════════════════════
#  主分割流程
# ═══════════════════════════════════════════════════════════


def segment_geometric(
    points: np.ndarray,
    params: GeometricParams | None = None,
    return_info: bool = False,
):
    """对单个构件点云做几何分割.

    Args:
        points: (N,3) 绝对坐标 (mm)
    Returns:
        labels (N,) int64; return_info=True 时返回 (labels, info).
    """
    p = params or GeometricParams()
    rng = np.random.default_rng(p.seed)
    pts = np.asarray(points, dtype=np.float64)

    # 0) 预处理 (只在点数超限时降采样; inv 把降采样云上的标签映射回全量点云)
    voxel = 0.0
    if len(pts) > p.max_points:
        voxel = p.voxel_mm * max(1.0, (len(pts) / p.max_points) ** (1 / 3))
    ds, inv = voxel_downsample(pts, voxel) if voxel > 0 else (pts, None)

    # 1) 局部几何特征 (固定物理半径邻域, 使阈值不随点云密度漂移)
    feat = local_pca_features(
        ds, p.knn, p.feature_chunk, radius_mm=p.feat_radius_mm
    )
    nrm = feat["normal"]

    # 2) 噪声
    noise = statistical_outlier_mask(feat["knn_dist"], p.sor_alpha)
    eligible = ~noise

    labels = np.full(len(ds), LABEL_CONCRETE, dtype=np.int64)
    labels[noise] = LABEL_NOISE
    info: dict = {"voxel_mm": voxel, "n_ds": len(ds), "n_input": len(pts)}

    # 3a) 套筒: 大直径圆柱 RANSAC (候选按曲率筛选, 接受模型半径限定在套筒量级)
    sleeve_cand = eligible & (feat["curvature"] >= p.sleeve_curv_min)
    cyls = ransac_cylinders(
        ds, nrm, sleeve_cand, p, rng, expect_radius=p.sleeve_radius_range
    )
    n_sleeve = 0
    for c in cyls:
        labels[c["inliers"] & eligible] = LABEL_SLEEVE
        n_sleeve += 1
    info["n_cyl_sleeve"] = n_sleeve
    info["cyl_radii"] = [round(c["radius"], 1) for c in cyls]

    # 3b) 钢筋: 经典 PCA 线性度判别 (点数过碎, 不足以做稳健圆柱拟合)
    rebar_mask = eligible & (feat["linearity"] >= p.lin_min) & (labels != LABEL_SLEEVE)
    labels[rebar_mask] = LABEL_REBAR
    info["n_rebar_pts"] = int(rebar_mask.sum())

    # 3c) 三分类模式: 金属 = 钢筋 ∪ 套筒, 与 Stage 1 同口径.
    #     实测表明几何方法给不出"套筒"这一类 (见 GeometricParams 中的实测说明),
    #     四分类口径下套筒 IoU 恒为 0; 只有在三分类口径下几何基线与 Stage 1 才可比.
    #     金属判据 = 线性度阈值 ∪ 已检出的套筒圆柱, 不再叠加曲率阈值 —— 曲率判据会把
    #     约 8% 的混凝土点误纳进金属类, 在 1.7% 的金属占比下精度会崩溃.
    if p.three_class:
        metal = eligible & ((feat["linearity"] >= p.lin_min) | (labels == LABEL_SLEEVE))
        labels[:] = LABEL_CONCRETE
        labels[noise] = LABEL_NOISE
        labels[metal] = LABEL_REBAR
        info["n_metal_pts"] = int(metal.sum())
        info["three_class"] = True
        info["frac_ds"] = {int(c): round(float((labels == c).mean()), 4) for c in range(4)}
        out = labels if inv is None else labels[inv]
        if return_info:
            info["frac_full"] = {
                int(c): round(float((out == c).mean()), 4) for c in range(4)
            }
            return out, info
        return out

    # 4) 平面 -> 混凝土
    rest = eligible & (labels == LABEL_CONCRETE)
    planes = ransac_planes(ds, rest, p, rng)
    on_plane = np.zeros(len(ds), dtype=bool)
    for pl in planes:
        on_plane |= pl["inliers"] & rest
    info["n_planes"] = len(planes)
    info["plane_tols"] = [round(pl["tol"], 2) for pl in planes]

    # 5) 兜底: 未落在任何平面上的非噪声点用邻域投票; 孤立者判为噪声
    labels[eligible & ~on_plane & (labels == LABEL_CONCRETE)] = -1
    todo = np.flatnonzero(labels == -1)
    if len(todo):
        tree = cKDTree(ds)
        kq = min(p.vote_k + 1, len(ds))
        _, idx = tree.query(ds[todo], k=kq, workers=-1)
        if idx.ndim == 1:
            idx = idx[:, None]
        nb = labels[idx]
        for a in range(len(todo)):
            vals = nb[a][nb[a] >= 0]
            if len(vals) == 0:
                labels[todo[a]] = LABEL_NOISE
            else:
                labels[todo[a]] = int(np.argmax(np.bincount(vals, minlength=4)))
    labels[noise] = LABEL_NOISE

    # 6) 传回全量点
    out = labels if inv is None else labels[inv]
    if return_info:
        info["frac_ds"] = {int(c): round(float((labels == c).mean()), 4) for c in range(4)}
        info["frac_full"] = {int(c): round(float((out == c).mean()), 4) for c in range(4)}
        return out, info
    return out


# ═══════════════════════════════════════════════════════════
#  指标 (与 lib/utils/metrics.py 数值等价, 便于无 torch 环境自检)
# ═══════════════════════════════════════════════════════════


def compute_iou_np(pred: np.ndarray, target: np.ndarray, class_names: dict) -> dict:
    """与 lib.utils.metrics.compute_iou 同口径 (同一 1e-6 eps 与键名)."""
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    ious: dict = {}
    for c, name in class_names.items():
        inter = float(((pred == c) & (target == c)).sum())
        union = float(((pred == c) | (target == c)).sum())
        ious[name] = inter / (union + 1e-6)
    vals = [v for k, v in ious.items() if k != "mIoU"]
    ious["mIoU"] = float(np.mean(vals)) if vals else 0.0
    return ious


def rs_iou_np(pred: np.ndarray, target: np.ndarray) -> float:
    """钢筋+套筒合并类 IoU (与 lib/cascade/validate.py 同式)."""
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    inter = float(((pred == 2) & (target == 2)).sum()) + float(
        ((pred == 3) & (target == 3)).sum()
    )
    union = float(((pred == 2) | (target == 2)).sum()) + float(
        ((pred == 3) | (target == 3)).sum()
    )
    return inter / (union + 1e-10)
