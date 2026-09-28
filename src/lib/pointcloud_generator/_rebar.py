"""
钢筋点云生成（纵向筋、箍筋、U形发夹筋）。
"""

import numpy as np
from typing import Optional

from ._config import (
    LABEL_REBAR, CONCRETE_COVER, REBAR_POINTS_PER_LENGTH,
    STIRRUP_SPACING_RANGE,
)


def generate_stirrups_rect(
    polygon: np.ndarray,
    z_min: float,
    z_max: float,
    stirrup_diam: float,
    concrete_cover: float,
    spacing: float,
    density: float,
    rng: np.random.Generator,
    protrusion: float = 0.0,
    axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    矩形截面箍筋生成（XY 平面闭环，沿轴向等距分布）。
    装配式梁的箍筋顶部可露出混凝土表面（protrusion > 0）。

    Parameters
    ----------
    polygon : ndarray, (4, 2) — 截面四角（截面平面坐标）。
    z_min, z_max : float
        箍筋沿轴向的范围（底部、顶部）。
    stirrup_diam : float
        箍筋直径（mm）。
    concrete_cover : float
        保护层厚度。
    spacing : float
        箍筋间距（mm）。
    density : float
    rng : np.random.Generator
    protrusion : float
        露出混凝土表面的长度（装配式梁特征）。
    axis : int
        轴向：2=Z（柱默认），0=X（梁）。控制最终点坐标的映射。
    """
    if len(polygon) < 3:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32)
    
    if protrusion > 0:
        z_top_actual = z_max + protrusion
    else:
        z_top_actual = z_max
    
    # 内缩多边形
    shrunk = _shrink_polygon(polygon, concrete_cover + stirrup_diam / 2)
    if shrunk is None or len(shrunk) < 3:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32)
    
    # 箍筋段：矩形四边闭环 (0→1, 1→2, 2→3, 3→0)
    segments = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    # 沿轴向等距生成箍筋
    n_stirrups = max(2, int((z_top_actual - z_min) / spacing) + 1)
    z_positions = np.linspace(z_min, z_top_actual, n_stirrups)
    
    all_pts = []
    n_per_segment = max(2, int(density * spacing * 0.1))
    
    for z_curr in z_positions:
        for seg in segments:
            i, j = seg
            if axis == 0:
                # X 轴向（梁）：输出 [X_轴向, Y_截面, Z_截面]
                start = np.array([z_curr, shrunk[i, 0], shrunk[i, 1]])
                end = np.array([z_curr, shrunk[j, 0], shrunk[j, 1]])
            else:
                # Z 轴向（柱默认）：输出 [X_截面, Y_截面, Z_轴向]
                start = np.array([shrunk[i, 0], shrunk[i, 1], z_curr])
                end = np.array([shrunk[j, 0], shrunk[j, 1], z_curr])
            t_vals = rng.uniform(0, 1, size=n_per_segment)
            pts = start + t_vals[:, np.newaxis] * (end - start)
            all_pts.append(pts)
    
    points = np.concatenate(all_pts) if all_pts else np.empty((0, 3))
    labels = np.full(len(points), LABEL_REBAR, dtype=np.int32)
    return points, labels


def generate_stirrups_polygon(
    polygon: np.ndarray,
    z_min: float,
    z_max: float,
    stirrup_diam: float,
    concrete_cover: float,
    spacing: float,
    density: float,
    rng: np.random.Generator,
    axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    通用多边形截面箍筋生成。
    沿多边形轮廓生成箍筋（适用于 L 形柱等非矩形截面）。
    """
    shrunk = _shrink_polygon(polygon, concrete_cover + stirrup_diam / 2)
    if shrunk is None or len(shrunk) < 3:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32)
    
    n_vertices = len(shrunk)
    segments = [(i, (i + 1) % n_vertices) for i in range(n_vertices)]
    
    n_stirrups = max(2, int((z_max - z_min) / spacing) + 1)
    z_positions = np.linspace(z_min, z_max, n_stirrups)
    
    all_pts = []
    n_per_segment = max(2, int(density * spacing * 0.1))
    
    for z_curr in z_positions:
        for seg in segments:
            i, j = seg
            if axis == 0:
                start = np.array([z_curr, shrunk[i, 0], shrunk[i, 1]])
                end = np.array([z_curr, shrunk[j, 0], shrunk[j, 1]])
            else:
                start = np.array([shrunk[i, 0], shrunk[i, 1], z_curr])
                end = np.array([shrunk[j, 0], shrunk[j, 1], z_curr])
            t_vals = rng.uniform(0, 1, size=n_per_segment)
            pts = start + t_vals[:, np.newaxis] * (end - start)
            all_pts.append(pts)
    
    points = np.concatenate(all_pts) if all_pts else np.empty((0, 3))
    labels = np.full(len(points), LABEL_REBAR, dtype=np.int32)
    return points, labels


def generate_u_hairpins(
    beam_length: float,
    beam_top_y: float,
    beam_bottom_y: float,
    concrete_top_z: float,
    diam: float,
    spacing_x: float,
    density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为梁上表面生成倒 U 形发夹筋（叠合梁特征）。

    发夹筋几何（YZ 截面，X 为长度方向）：
      顶部弯钩  ===（水平圆柱，连接两直腿顶部）
      左直腿    |     右直腿     |（垂直圆柱，立在混凝土表面）
    混凝土表面 ================

    参数：
      beam_length: 梁长 (mm)
      beam_top_y, beam_bottom_y: 梁顶/底 Y 坐标（确定开口宽度）
      concrete_top_z: 混凝土表面 Z 坐标
      diam: 发夹筋直径 (mm)
      spacing_x: 沿梁长间距 (mm)
      density: 采样密度
      rng: 随机数生成器
    """
    radius = diam / 2.0
    y_range = abs(beam_top_y - beam_bottom_y)
    gap = max(40.0, min(y_range * 0.85, y_range - 30.0))
    hairpin_height = min(150.0, max(100.0, gap * 0.5))
    mid_y = (beam_top_y + beam_bottom_y) / 2.0
    y_left = mid_y - gap / 2.0
    y_right = mid_y + gap / 2.0
    n_hairpins = max(1, int(beam_length / spacing_x))
    end_spacing = spacing_x * 0.6
    mid_spacing = spacing_x * 1.4
    transition = beam_length * 0.25

    from ._geometry import sample_cylinder_segment

    pts_list = []
    x_pos = transition * 0.2
    while x_pos < beam_length - end_spacing * 0.3:
        if x_pos < transition or x_pos > beam_length - transition:
            sp = end_spacing
        else:
            sp = mid_spacing
        z_base = concrete_top_z
        z_top = concrete_top_z + hairpin_height
        p_lb = np.array([x_pos, y_left, z_base])
        p_lt = np.array([x_pos, y_left, z_top])
        p_rb = np.array([x_pos, y_right, z_base])
        p_rt = np.array([x_pos, y_right, z_top])
        pts_list.append(sample_cylinder_segment(p_lb, p_lt, radius, density, rng))
        pts_list.append(sample_cylinder_segment(p_rb, p_rt, radius, density, rng))
        pts_list.append(sample_cylinder_segment(p_lt, p_rt, radius, density, rng))
        x_pos += sp

    points = np.vstack(pts_list) if pts_list else np.empty((0, 3))
    labels = np.full(len(points), LABEL_REBAR, dtype=np.int32)
    return points, labels


def _shrink_polygon(polygon: np.ndarray, offset: float) -> Optional[np.ndarray]:
    """
    多边形向内收缩 offset 距离。
    使用 shapely buffer 实现。
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    poly = ShapelyPolygon(polygon)
    if not poly.is_valid:
        return None
    shrunk = poly.buffer(-offset, resolution=8)
    if shrunk.is_empty or not shrunk.is_valid:
        return None
    ext = np.array(shrunk.exterior.coords)[:-1]  # 去掉首尾重复
    return ext


def _points_along_polygon(
    polygon: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    沿多边形边均匀取 n_points 个点。
    """
    if len(polygon) < 3:
        return None
    # 计算边长
    verts = np.vstack([polygon, polygon[0]])
    edges = np.diff(verts, axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    total_length = edge_lengths.sum()
    
    # 按边长比例分配点数
    n_per_edge = np.maximum(1, (n_points * edge_lengths / total_length).astype(int))
    
    # 修正总点数
    diff = n_points - n_per_edge.sum()
    if diff > 0:
        n_per_edge[np.argmax(edge_lengths)] += diff
    elif diff < 0:
        n_per_edge[np.argmin(edge_lengths)] += diff
    
    all_pts = []
    for i in range(len(polygon)):
        n = n_per_edge[i]
        if n <= 0:
            continue
        t = np.linspace(0, 1, n, endpoint=False)
        pt = verts[i] + t[:, np.newaxis] * edges[i]
        all_pts.append(pt)
    
    return np.concatenate(all_pts) if all_pts else None
