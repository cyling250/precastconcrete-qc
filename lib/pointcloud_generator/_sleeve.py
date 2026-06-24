"""
灌浆套筒点云生成模块。

套筒被预埋在混凝土端部，与纵向钢筋同轴对齐。
激光扫描只能探测到端口圆孔、内壁和底部，外壁埋在混凝土中不可见。

密度策略（与基准密度 D 对比）：
  - 端口圆孔：混凝土表面减去圆形区域（无点）
  - 内壁采样：D × SLEEVE_INNER_DENSITY_FACTOR (0.5)
  - 底部采样：D × SLEEVE_BOTTOM_DENSITY_FACTOR (0.7)
"""

import numpy as np
from typing import Optional

from ._config import (
    LABEL_SLEEVE, LABEL_CONCRETE,
    SLEEVE_INNER_DENSITY_FACTOR, SLEEVE_BOTTOM_DENSITY_FACTOR,
)
from ._geometry import sample_cylinder_side, sample_disk, sample_circle_perimeter


def generate_sleeve(
    end_center_xy: tuple[float, float],
    end_z: float,
    inner_diam: float,
    depth: float,
    density: float,
    rng: np.random.Generator,
    direction: int = 1,
    axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成一个灌浆套筒的点云（内壁 + 底部）。

    套筒轴线沿 axis 方向从端面向构件内部延伸 depth。
    端口圆孔在端面处，圆孔由调用方（混凝土模块）处理。

    Parameters
    ----------
    end_center_xy : (c1, c2)
        套筒轴线在截面平面上的位置（对应纵筋位置）。
    end_z : float
        套筒端口端面坐标（沿轴线）。
    inner_diam : float
        套筒内径 (mm)。
    depth : float
        套筒深度 (mm)，即沿轴线的长度。
    density : float
        基准采样密度 (pts/mm²)。
    rng : np.random.Generator
    direction : int
        端口朝向：+1 正向，-1 负向。
    axis : int
        套筒轴向：0=X, 1=Y, 2=Z（默认 Z，用于柱；梁使用 0=X）。
    
    Returns
    -------
    points : ndarray, (N, 3)
    labels : ndarray, (N,) — 全为 LABEL_SLEEVE
    """
    cx, cy = end_center_xy
    radius = inner_diam / 2.0
    
    # 套筒从端口向构件内部延伸
    z_bottom = end_z - direction * depth
    z_min = min(end_z, z_bottom)
    z_max = max(end_z, z_bottom)
    
    # ─── 内壁（圆柱侧面）───
    inner_density = density * SLEEVE_INNER_DENSITY_FACTOR
    inner_surface_area = 2 * np.pi * radius * depth
    n_inner = max(4, int(inner_surface_area * inner_density))
    
    wall_pts = sample_cylinder_side(radius, depth, n_inner, rng)
    
    if axis == 0:
        # X 轴向（梁）：圆柱沿 X，截面在 YZ
        # sample_cylinder_side 返回 (x,y,z)，其中 z 是轴向
        # 交换：z→x, x→y, y→z
        pts_3d = np.column_stack([
            wall_pts[:, 2] + (z_min + z_max) / 2,  # X = 原 Z (轴向)
            wall_pts[:, 0] + cx,                     # Y
            wall_pts[:, 1] + cy,                     # Z
        ])
    elif axis == 1:
        # Y 轴向
        pts_3d = np.column_stack([
            wall_pts[:, 0] + cx,
            wall_pts[:, 2] + (z_min + z_max) / 2,
            wall_pts[:, 1] + cy,
        ])
    else:
        # Z 轴向（柱，默认）
        pts_3d = wall_pts
        pts_3d[:, 0] += cx
        pts_3d[:, 1] += cy
        pts_3d[:, 2] += (z_min + z_max) / 2
    
    # ─── 底部（圆盘）───
    bottom_density = density * SLEEVE_BOTTOM_DENSITY_FACTOR
    bottom_area = np.pi * radius ** 2
    n_bottom = max(4, int(bottom_area * bottom_density))
    
    bottom_pts = sample_disk(radius, n_bottom, rng)
    # bottom_pts 是 (N, 3) 在 XY 平面 z=0 的圆盘
    
    if axis == 0:
        # X 轴向：圆盘在 YZ 平面
        bottom_transformed = np.column_stack([
            np.full(n_bottom, z_bottom),
            bottom_pts[:, 0] + cx,
            bottom_pts[:, 1] + cy,
        ])
    elif axis == 1:
        bottom_transformed = np.column_stack([
            bottom_pts[:, 0] + cx,
            np.full(n_bottom, z_bottom),
            bottom_pts[:, 1] + cy,
        ])
    else:
        bottom_transformed = bottom_pts
        bottom_transformed[:, 0] += cx
        bottom_transformed[:, 1] += cy
        bottom_transformed[:, 2] = z_bottom
    
    # ─── 合并 ───
    points = np.vstack([pts_3d, bottom_transformed])
    labels = np.full(len(points), LABEL_SLEEVE, dtype=np.int32)
    return points, labels


def generate_sleeves_for_component(
    rebar_positions_xy: list[tuple[float, float]],
    end_z: float,
    inner_diam: float,
    depth: float,
    density: float,
    rng: np.random.Generator,
    direction: int = 1,
    axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为构件一端的所有纵筋位置生成套筒。

    Parameters
    ----------
    rebar_positions_xy : list of (cx, cy)
        每根纵筋在截面上的位置（套筒与之对齐）。
    end_z : float
        端面坐标（沿轴线）。
    inner_diam : float
        套筒内径。
    depth : float
        套筒深度。
    density : float
        基准密度。
    rng : np.random.Generator
    direction : int
    axis : int
        轴向：0=X, 1=Y, 2=Z
        
    Returns
    -------
    points : ndarray, (N, 3)
    labels : ndarray, (N,)
    """
    all_pts, all_lbls = [], []
    for pos in rebar_positions_xy:
        pts, lbls = generate_sleeve(
            end_center_xy=pos,
            end_z=end_z,
            inner_diam=inner_diam,
            depth=depth,
            density=density,
            rng=rng,
            direction=direction,
            axis=axis,
        )
        all_pts.append(pts)
        all_lbls.append(lbls)
    
    if not all_pts:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32)
    
    return np.vstack(all_pts), np.concatenate(all_lbls)


def get_sleeve_hole_circles(
    rebar_positions_xy: list[tuple[float, float]],
    inner_diam: float,
) -> list[tuple[tuple[float, float], float]]:
    """
    返回套筒端口圆孔的 (中心, 半径) 列表。
    供混凝土模块在端面上挖孔使用。
    
    Parameters
    ----------
    rebar_positions_xy : list of (cx, cy)
    inner_diam : float
        
    Returns
    -------
    list of ((cx, cy), radius)
    """
    radius = inner_diam / 2.0
    return [(pos, radius) for pos in rebar_positions_xy]
