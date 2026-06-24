"""
基础几何采样工具（纯数学，不做工程假设）。
提供各种几何体的表面点云采样函数。
"""

import numpy as np
from typing import Optional


def sample_rectangle_surface(
    width: float,
    height: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    在矩形面内均匀采点。

    Parameters
    ----------
    width, height : float
        矩形的宽和高（mm）。
    density : float
        采样密度（pts/mm²）。
    rng : np.random.Generator

    Returns
    -------
    points : ndarray, shape (N, 2), 二维平面坐标。
    """
    area = width * height
    n_points = max(4, int(area * density))
    xs = rng.uniform(-width / 2, width / 2, size=n_points)
    ys = rng.uniform(-height / 2, height / 2, size=n_points)
    return np.column_stack([xs, ys])


def sample_cylinder_surface(
    radius: float,
    height: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    圆柱侧面均匀采样。

    Parameters
    ----------
    radius : float
        圆柱半径（mm）。
    height : float
        高度（mm）。
    density : float
        采样密度（pts/mm²）。
    rng : np.random.Generator

    Returns
    -------
    points : ndarray, shape (N, 3).
    """
    surface_area = 2 * np.pi * radius * height
    n_points = max(4, int(surface_area * density))
    theta = rng.uniform(0, 2 * np.pi, size=n_points)
    z = rng.uniform(-height / 2, height / 2, size=n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def sample_cylinder_side(
    radius: float,
    height: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    圆柱侧面采指定数量点。
    """
    theta = np.linspace(0, 2 * np.pi, max(12, n_points), endpoint=False)
    z_vals = rng.uniform(-height / 2, height / 2, size=len(theta))
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, z_vals])


def sample_cylinder_segment(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    在两个三维点之间生成圆柱侧面采样点。

    Parameters
    ----------
    start, end : (3,) 端点坐标。
    radius : 半径 (mm)。
    density : 采样密度 (pts/mm²)。
    rng : 随机数生成器。

    Returns
    -------
    points : (N, 3)
    """
    direction = end - start
    length_val = np.linalg.norm(direction)
    if length_val < 1e-8:
        return np.empty((0, 3))
    dir_norm = direction / length_val
    # 构造正交基
    if abs(dir_norm[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])
    u -= np.dot(u, dir_norm) * dir_norm
    u /= np.linalg.norm(u)
    v = np.cross(dir_norm, u)
    surface_area = 2 * np.pi * radius * length_val
    n_pts = max(4, int(surface_area * density))
    theta = rng.uniform(0, 2 * np.pi, size=n_pts)
    t = rng.uniform(0, length_val, size=n_pts)
    pts = (
        start[None, :]
        + t[:, None] * dir_norm[None, :]
        + radius
        * (np.cos(theta)[:, None] * u[None, :] + np.sin(theta)[:, None] * v[None, :])
    )
    return pts


def sample_disk(
    radius: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    圆盘底面均匀采样（使用拒绝采样法）。
    """
    points = []
    while len(points) < n_points:
        xs = rng.uniform(-radius, radius, size=n_points * 2)
        ys = rng.uniform(-radius, radius, size=n_points * 2)
        radii = np.sqrt(xs**2 + ys**2)
        mask = radii <= radius
        valid = np.column_stack([xs[mask], ys[mask], np.zeros(mask.sum())])
        points.append(valid)
    return np.concatenate(points)[:n_points]


def sample_circle_perimeter(
    radius: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    圆环（圆周）上均匀采点，用于套筒端口边缘的离散化。
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def sample_polygon_surface(
    polygon: np.ndarray,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    任意多边形内部均匀采样，使用拒绝采样+向量化判断。

    Parameters
    ----------
    polygon : ndarray, shape (M, 2)
        多边形顶点（逆时针，首尾相接或不接均可）。
    density : float
        采样密度（pts/mm²）。
    rng : np.random.Generator

    Returns
    -------
    points : ndarray, shape (N, 2).
    """
    from shapely.geometry import Polygon as ShapelyPolygon, Point
    import shapely.vectorized as sv

    poly = ShapelyPolygon(polygon)
    if not poly.is_valid or poly.area <= 0:
        return np.empty((0, 2))

    area = poly.area
    n_points = max(4, int(area * density))
    if n_points > 1_000_000:
        n_points = 1_000_000

    min_x, min_y, max_x, max_y = poly.bounds

    # 批量拒绝采样：每次生成 batch_size × 2 个候选点
    batch_size = min(max(n_points * 3, 1000), 200000)
    collected = []
    total_needed = n_points

    while total_needed > 0:
        xs = rng.uniform(min_x, max_x, size=batch_size)
        ys = rng.uniform(min_y, max_y, size=batch_size)
        # 向量化判断
        mask = sv.contains(poly, xs, ys)
        valid_x = xs[mask][:total_needed]
        valid_y = ys[mask][:total_needed]
        collected.extend(zip(valid_x.tolist(), valid_y.tolist()))
        total_needed -= len(valid_x)

    return np.array(collected[:n_points])


def sample_polygon_prism_surface(
    polygon: np.ndarray,
    height: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    多边形棱柱表面采样（侧面 + 顶面 + 底面）。

    Parameters
    ----------
    polygon : ndarray, shape (M, 2)
        横截面多边形顶点（XY 平面）。
    height : float
        棱柱高度（Z 方向）。
    density : float
        采样密度（pts/mm²）。
    rng : np.random.Generator

    Returns
    -------
    points : ndarray, shape (N, 3).
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    poly = ShapelyPolygon(polygon)
    area = poly.area
    perimeter = poly.length

    # 侧面点数
    side_area = perimeter * height
    n_side = max(4, int(side_area * density))

    # 顶底点数
    n_top = max(4, int(area * density * 0.5))
    n_bottom = n_top

    # 侧面采样：沿多边形边 + Z 方向
    vertices = polygon
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    edge_lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    total_perimeter = edge_lengths.sum()

    side_pts = []
    if n_side > 0:
        # 按边长比例分配采样点
        for i in range(len(vertices) - 1):
            n_edge = max(1, int(n_side * edge_lengths[i] / total_perimeter))
            t_vals = rng.uniform(0, 1, size=n_edge)
            z_vals = rng.uniform(-height / 2, height / 2, size=n_edge)
            x = vertices[i, 0] + t_vals * (vertices[i + 1, 0] - vertices[i, 0])
            y = vertices[i, 1] + t_vals * (vertices[i + 1, 1] - vertices[i, 1])
            side_pts.append(np.column_stack([x, y, z_vals]))
        side_pts = np.concatenate(side_pts) if side_pts else np.empty((0, 3))
    else:
        side_pts = np.empty((0, 3))

    # 顶面
    top_2d = sample_polygon_surface(polygon, density * 0.5, rng)
    top_pts = np.column_stack([top_2d, np.full(len(top_2d), height / 2)])

    # 底面
    bottom_2d = sample_polygon_surface(polygon, density * 0.5, rng)
    bottom_pts = np.column_stack([bottom_2d, np.full(len(bottom_2d), -height / 2)])

    return (
        np.vstack([side_pts, top_pts, bottom_pts])
        if len(side_pts) > 0
        else np.vstack([top_pts, bottom_pts])
    )


def sample_line_segment(
    start: np.ndarray,
    end: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    线段上均匀采点。
    """
    t = rng.uniform(0, 1, size=n_points)
    return start + t[:, np.newaxis] * (end - start)


def sample_circle_arc(
    center: np.ndarray,
    radius: float,
    start_angle: float,
    end_angle: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    圆弧上均匀采点。
    """
    angles = rng.uniform(start_angle, end_angle, size=n_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])
