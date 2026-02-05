"""
点云预处理工具函数
"""

import open3d as o3d
import numpy as np


def desample_point_cloud(
    pcd: o3d.geometry.PointCloud, voxel_size: float = 0.05
) -> o3d.geometry.PointCloud:
    """
    对点云进行体素下采样。

    参数:
    pcd (o3d.geometry.PointCloud): 输入的点云对象。
    voxel_size (float): 体素大小，默认值为0.005。

    返回:
    o3d.geometry.PointCloud: 下采样后的点云对象。
    """
    return pcd.voxel_down_sample(voxel_size)


def _distance(pcd: o3d.geometry.PointCloud, point: np.ndarray) -> np.ndarray:
    """
    计算点云中点到给定点的最近距离。

    参数:
    pcd (o3d.geometry.PointCloud): 输入的点云对象。
    point (np.ndarray): 给定的点，形状为(3,)。

    返回:
    np.ndarray: 点云中点到给定点的最近距离，形状为(n_points,)。

    异常:
    ValueError: 如果点云为空或输入点无效。
    """
    if len(pcd.points) == 0:
        raise ValueError("点云为空")

    if not isinstance(point, np.ndarray) or point.shape != (3,):
        raise ValueError("输入点必须是形状为(3,)的NumPy数组")

    # 使用KDTreeFlann进行更高效的最近邻搜索
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    _, indices, distances = kdtree.search_knn_vector_3d(point, 1)

    return distances
