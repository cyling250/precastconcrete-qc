"""
混凝土表面点云生成。
为各种截面形状的混凝土构件提供表面采样函数。
"""

import numpy as np
import open3d as o3d
from typing import Optional

from ._config import (
    LABEL_CONCRETE, CONCRETE_POINTS_PER_AREA,
)
from ._geometry import sample_polygon_surface


def _apply_surface_roughness(
    pts: np.ndarray,
    roughness: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """沿法线方向添加表面粗糙度噪声。

    使用 Open3D 估计每点的法线方向，沿法线添加高斯扰动。
    适用于各种形状的混凝土表面。

    Parameters
    ----------
    pts : ndarray, (N, 3)
    roughness : float
        粗糙度 σ (mm)，高斯扰动的标准差。
    rng : np.random.Generator

    Returns
    -------
    ndarray, (N, 3)
    """
    if roughness <= 0 or len(pts) == 0:
        return pts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(30)
    )
    normals = np.asarray(pcd.normals)

    # 各向同性高斯扰动沿法线方向
    displacement = rng.normal(0, roughness, size=len(pts))
    return pts + normals * displacement[:, np.newaxis]


def sample_concrete_prism(
    polygon: np.ndarray,
    height: float,
    density: float,
    rng: np.random.Generator,
    roughness: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    混凝土棱柱表面采样（侧面 + 顶面 + 底面）。
    
    Parameters
    ----------
    polygon : ndarray, shape (M, 2)
        横截面多边形顶点（XY 平面）。
    height : float
        棱柱高度（Z 方向）。
    density : float
        采样密度（pts/mm²）。
    rng : np.random.Generator
    roughness : float, optional
        表面粗糙度 σ (mm)。为 None 时不添加粗糙度。
        
    Returns
    -------
    points : ndarray, (N, 3)
    labels : ndarray, (N,) — 全为 LABEL_CONCRETE
    """
    from ._geometry import sample_polygon_prism_surface
    
    # 完整棱柱表面采样
    pts = sample_polygon_prism_surface(polygon, height, density, rng)
    
    # 表面粗糙度模拟 — 沿法线方向添加噪声
    if roughness is not None and roughness > 0:
        pts = _apply_surface_roughness(pts, roughness, rng)
    
    labels = np.full(len(pts), LABEL_CONCRETE, dtype=np.int32)
    return pts, labels


def generate_concrete_face_grid(
    polygon: np.ndarray,
    z_level: float,
    density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成混凝土截面（特定 Z 高度处的多边形面）的点云。
    用于构件端面等。
    
    Parameters
    ----------
    polygon : ndarray, shape (M, 2)
    z_level : float
        截面 Z 坐标。
    density : float
    rng : np.random.Generator
        
    Returns
    -------
    points : ndarray, (N, 3)
    labels : ndarray, (N,)
    """
    pts_2d = sample_polygon_surface(polygon, density, rng)
    pts_3d = np.column_stack([pts_2d, np.full(len(pts_2d), z_level)])
    labels = np.full(len(pts_3d), LABEL_CONCRETE, dtype=np.int32)
    return pts_3d, labels


def remove_circular_hole(
    points: np.ndarray,
    labels: np.ndarray,
    center_xy: tuple[float, float],
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从混凝土点云中移除圆形区域内的点（用于套筒端口空洞）。
    
    Parameters
    ----------
    points : ndarray, (N, 3)
    labels : ndarray, (N,)
    center_xy : (cx, cy)
    radius : float
        挖孔半径。
    
    Returns
    -------
    points_filtered : ndarray, (M, 3)
    labels_filtered : ndarray, (M,)
    """
    cx, cy = center_xy
    dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    mask = dists >= radius
    return points[mask], labels[mask]
