"""
综合多级噪声增强 v2.

主要改进 (相对 v1):
  1. 矩形遮挡 → 随机多边形遮挡 (3~7 边形), 模拟真实遮挡形状
  2. 非均匀密度: 对构件做 2~4 个空间分区, 每区密度不同 (模拟扫描距离变化)
  3. 结构性噪点簇: 在构件表面附近生成高斯噪点簇 (模拟堆料/杂物)
  4. 边缘噪点: 类别边界处添加混合像素和抖动 (模拟激光扫描边缘效应)
  5. 全面提高外来噪点比例
"""

import numpy as np

from matplotlib.path import Path
from scipy.spatial import KDTree

from ._config import (
    LABEL_NOISE,
    DENSITY_VARIATION_N_ZONES,
    DENSITY_VARIATION_FACTOR_RANGE,
    CLUSTER_OUTLIER_N_CLUSTERS,
    CLUSTER_OUTLIER_SPREAD,
    EDGE_NOISE_SEARCH_RADIUS,
    EDGE_NOISE_JITTER_SIGMA,
    EDGE_NOISE_LABEL_FLIP_PROB,
)


def apply_noise(
    points: np.ndarray,
    labels: np.ndarray,
    noise_config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    综合多级噪声增强（一次性叠加全部噪声类型）.

    所有噪声参数均为范围值, 对每个构件独立采样 (rng 已按构件种子).
    应用顺序: 遮挡 → 均匀噪点 → 聚类噪点 → 边缘噪点 → 丢弃 → 抖动
    """
    # 从范围中采样具体参数 (每个构件不一样)
    occ_range = noise_config.get("occlusion_pct_range", (0, 0))
    occlusion_pct = rng.uniform(*occ_range)

    jit_range = noise_config.get("jitter_sigma_range", (0, 0))
    jitter_sigma = rng.uniform(*jit_range)

    keep_range = noise_config.get("keep_pct_range", (1, 1))
    keep_pct = rng.uniform(*keep_range)

    out_range = noise_config.get("outlier_pct_range", (0, 0))
    outlier_pct = rng.uniform(*out_range)

    clus_range = noise_config.get("cluster_outlier_pct_range", (0, 0))
    cluster_pct = rng.uniform(*clus_range)

    edge_range = noise_config.get("edge_noise_pct_range", (0, 0))
    edge_pct = rng.uniform(*edge_range)

    n_occ_poly = noise_config.get("n_occlusion_polygons", (1, 3))
    n_occ_sides = noise_config.get("n_occlusion_sides", (3, 7))

    pts, lbls = points.copy(), labels.copy()

    # 1) 多边形遮挡
    if occlusion_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_polygon_occlusion(pts, lbls, occlusion_pct, n_occ_poly, n_occ_sides, rng)

    # 2) 均匀外来噪点
    if outlier_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_outliers(pts, lbls, outlier_pct, rng)

    # 3) 结构性噪点簇
    if cluster_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_clustered_outliers(pts, lbls, cluster_pct, rng)

    # 4) 边缘噪点
    if edge_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_edge_noise(pts, lbls, edge_pct, rng)

    # 5) 随机丢弃
    if keep_pct < 1.0 and len(pts) > 0:
        pts, lbls = _apply_dropout(pts, lbls, keep_pct, rng)

    # 6) 高斯抖动
    if jitter_sigma > 0 and len(pts) > 0:
        pts = _apply_jitter(pts, jitter_sigma, rng)

    return pts, lbls


# ═══════════════════════════════════════════════════════════
#  非均匀密度 (模拟扫描距离变化)
# ═══════════════════════════════════════════════════════════


def apply_density_variation(
    points: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对构件点云施加非均匀密度变化。

    将空间划分为 2~4 个 Voronoi 区域, 每区分配不同的保留系数 [0.75, 1.25],
    模拟激光扫描中不同部位因距离差异导致的密度变化.

    密度差别控制在 ±25% 以内, 同一构件最多 4 种密度.
    """
    N = len(points)
    if N < 2000:
        return points, labels

    n_zones = rng.integers(
        DENSITY_VARIATION_N_ZONES[0],
        DENSITY_VARIATION_N_ZONES[1] + 1,
    )
    factor_range = DENSITY_VARIATION_FACTOR_RANGE

    # 1) 随机选种子点 (Voronoi 分区)
    seed_indices = rng.choice(N, size=n_zones, replace=False)
    seeds = points[seed_indices]

    # 2) 每个点 -> 最近种子 (快速 Voronoi)
    tree = KDTree(seeds)
    zone_ids = tree.query(points)[1]

    # 3) 每区随机保留系数, 确保均值 ≈ 1.0
    factors = rng.uniform(factor_range[0], factor_range[1], size=n_zones)
    factors = factors / factors.mean()

    # 4) 逐区降采样
    keep_mask = np.zeros(N, dtype=bool)
    for z in range(n_zones):
        zone_mask = (zone_ids == z)
        n_zone = zone_mask.sum()
        if n_zone == 0:
            continue
        n_keep = max(1, min(n_zone, int(n_zone * factors[z])))
        keep_idx = rng.choice(np.where(zone_mask)[0], size=n_keep, replace=False)
        keep_mask[keep_idx] = True

    return points[keep_mask], labels[keep_mask]


# ═══════════════════════════════════════════════════════════
#  多边形遮挡 (替代矩形块)
# ═══════════════════════════════════════════════════════════


def _random_convex_polygon(
    center_xy: np.ndarray,
    max_radius: float,
    n_sides: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成随机凸多边形顶点 (N, 2), 顶点按逆时针排列."""
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_sides))
    radii = max_radius * rng.uniform(0.4, 1.0, n_sides)
    vertices = np.column_stack([
        center_xy[0] + radii * np.cos(angles),
        center_xy[1] + radii * np.sin(angles),
    ])
    return vertices


def _apply_polygon_occlusion(
    points: np.ndarray,
    labels: np.ndarray,
    occlusion_pct: float,
    n_polygons_range: tuple[int, int],
    n_sides_range: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    多边形遮挡: 用随机凸多边形棱柱 (沿 Z 轴拉伸) 遮挡部分点云.

    比矩形块更贴近真实遮挡形状 (如脚手架、模板支撑杆).
    """
    if len(points) == 0:
        return points, labels

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    max_radius = np.min(bbox_size[:2]) * 0.3

    n_polygons = rng.integers(n_polygons_range[0], n_polygons_range[1] + 1)
    z_min, z_max = bbox_min[2], bbox_max[2]
    mask = np.ones(len(points), dtype=bool)

    for _ in range(n_polygons):
        n_sides = rng.integers(n_sides_range[0], n_sides_range[1] + 1)
        center_xy = bbox_min[:2] + bbox_size[:2] * rng.uniform(0.1, 0.9, size=2)

        vertices = _random_convex_polygon(center_xy, max_radius, n_sides, rng)
        poly_path = Path(vertices, closed=True)

        # Z 轴拉伸 (棱柱)
        poly_z_min = rng.uniform(z_min, z_min + bbox_size[2] * 0.6)
        poly_z_max = rng.uniform(poly_z_min + bbox_size[2] * 0.2, z_max)

        in_xy = poly_path.contains_points(points[:, :2])
        in_z = (points[:, 2] >= poly_z_min) & (points[:, 2] <= poly_z_max)
        mask &= ~(in_xy & in_z)

    return points[mask], labels[mask]


# ═══════════════════════════════════════════════════════════
#  结构性噪点簇
# ═══════════════════════════════════════════════════════════


def _apply_clustered_outliers(
    points: np.ndarray,
    labels: np.ndarray,
    cluster_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    结构性噪点簇: 在构件表面附近生成高斯噪点簇.

    模拟施工现场堆料、模板、脚手架等离散在构件周围的杂散物体.
    与均匀噪点不同, 噪点簇在空间上聚集, 更难被模型忽略.
    """
    n_clusters = rng.integers(
        CLUSTER_OUTLIER_N_CLUSTERS[0],
        CLUSTER_OUTLIER_N_CLUSTERS[1] + 1,
    )
    n_outliers = max(1, int(len(points) * cluster_pct))

    # 在构件表面随机选取簇中心
    surface_pts = points[labels != LABEL_NOISE]
    if len(surface_pts) < n_clusters:
        surface_pts = points
    cluster_centers_idx = rng.choice(len(surface_pts), size=n_clusters, replace=False)
    cluster_centers = surface_pts[cluster_centers_idx]

    # 每簇点数 (按比例分配)
    ratios = rng.dirichlet(np.ones(n_clusters))
    counts = (np.array([int(n_outliers * r) for r in ratios[:-1]] + [max(1, n_outliers - int(n_outliers * ratios[:-1].sum()))]))

    outliers_list = []
    for center, count in zip(cluster_centers, counts):
        if count <= 0:
            continue
        spread = rng.uniform(CLUSTER_OUTLIER_SPREAD[0], CLUSTER_OUTLIER_SPREAD[1])
        # 各向异性: 不同轴不同散布
        spread_xyz = spread * rng.uniform(0.6, 1.4, size=3)
        cluster_pts = rng.normal(loc=center, scale=spread_xyz, size=(count, 3))
        outliers_list.append(cluster_pts)

    if not outliers_list:
        return points, labels

    outliers = np.vstack(outliers_list)
    pts_out = np.vstack([points, outliers])
    lbls_out = np.concatenate([
        labels,
        np.full(len(outliers), LABEL_NOISE, dtype=np.int32),
    ])
    return pts_out, lbls_out


# ═══════════════════════════════════════════════════════════
#  边缘噪点
# ═══════════════════════════════════════════════════════════


def _apply_edge_noise(
    points: np.ndarray,
    labels: np.ndarray,
    edge_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    边缘噪点: 在类别边界处添加混合像素效应.

    模拟激光扫描中边缘处的混合像素 (mixed pixel) 现象:
      - 边界处的点被不同类别的标签污染
      - 边界处的点坐标被额外抖动
    """
    N = len(points)
    if N < 100:
        return points, labels

    search_radius = EDGE_NOISE_SEARCH_RADIUS

    # 用 KDTree 找每个点邻域内是否有多类标签
    tree = KDTree(points)
    # 只查表面点 (非噪声类)
    surface_mask = labels != LABEL_NOISE
    surface_idx = np.where(surface_mask)[0]

    if len(surface_idx) < 10:
        return points, labels

    edge_mask = np.zeros(N, dtype=bool)
    for idx in surface_idx:
        neighbor_indices = tree.query_ball_point(points[idx], r=search_radius)
        neighbor_labels = labels[neighbor_indices]
        # 如果邻域内有不同类别的点 → 边缘点
        unique_labels = np.unique(neighbor_labels)
        if len(unique_labels) > 1:
            edge_mask[idx] = True

    edge_indices = np.where(edge_mask)[0]
    n_edge = len(edge_indices)
    if n_edge == 0:
        return points, labels

    # 选一部分边缘点添加噪声
    n_corrupt = max(1, int(n_edge * edge_pct))
    corrupt_idx = rng.choice(edge_indices, size=n_corrupt, replace=False)

    noisy_points = points.copy()
    noisy_labels = labels.copy()

    for idx in corrupt_idx:
        # 1) 坐标抖动: 混合像素导致的位置偏移
        jitter = rng.normal(0, EDGE_NOISE_JITTER_SIGMA, size=3)
        noisy_points[idx] += jitter

        # 2) 标签翻转: 以一定概率改成邻域内的另一类标签
        if rng.random() < EDGE_NOISE_LABEL_FLIP_PROB:
            neighbor_indices = tree.query_ball_point(points[idx], r=search_radius)
            neighbor_labels = labels[neighbor_indices]
            other_labels = neighbor_labels[neighbor_labels != labels[idx]]
            if len(other_labels) > 0:
                noisy_labels[idx] = rng.choice(other_labels)

    return noisy_points, noisy_labels


# ═══════════════════════════════════════════════════════════
#  原有噪声 (保留不变)
# ═══════════════════════════════════════════════════════════


def _apply_dropout(
    points: np.ndarray,
    labels: np.ndarray,
    keep_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """随机丢弃: 随机保留 keep_pct 比例的点."""
    n = len(points)
    n_keep = max(1, int(n * keep_pct))
    indices = rng.choice(n, size=n_keep, replace=False)
    return points[indices], labels[indices]


def _apply_jitter(
    points: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """高斯抖动: 沿每个坐标轴独立添加高斯噪声."""
    noise = rng.normal(0, sigma, size=points.shape)
    return points + noise


def _apply_outliers(
    points: np.ndarray,
    labels: np.ndarray,
    outlier_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """外来噪点: 在点云包围盒内均匀分布随机点."""
    n_outliers = max(1, int(len(points) * outlier_pct))
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    outliers = rng.uniform(bbox_min, bbox_max, size=(n_outliers, 3))
    pts_out = np.vstack([points, outliers])
    lbls_out = np.concatenate([
        labels,
        np.full(n_outliers, LABEL_NOISE, dtype=np.int32),
    ])
    return pts_out, lbls_out
