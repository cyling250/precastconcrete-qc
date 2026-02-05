"""
点云预处理工具函数
"""

import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from geometry.line import SegmentLine
from geometry.point import BasePoint


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


def calculate_obb_inscribed_cylinder_segment(
    points: np.ndarray, main_direction: np.ndarray = None
) -> tuple[SegmentLine, float]:
    """
    计算点云的OBB包围盒内接圆柱体的基线线段和半径。

    参数:
    points (np.ndarray): 输入的点云数据，形状为(n, 3)
    main_direction (np.ndarray, optional): 已知的主方向向量，形状为(3,)

    返回:
    tuple[SegmentLine, float]: 包含SegmentLine对象和圆柱体半径的元组

    异常:
    ValueError: 如果点云为空或点数不足
    """
    if len(points) == 0:
        raise ValueError("点云为空")

    if len(points) < 3:
        raise ValueError("点云点数不足，无法计算OBB")

    # 计算点云的协方差矩阵
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    covariance_matrix = np.dot(centered_points.T, centered_points) / len(
        centered_points
    )

    # 特征值分解，获取主方向
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 按特征值从大到小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 如果提供了已知主方向，确保与计算的主方向一致
    if main_direction is not None:
        dot_product = np.dot(eigenvectors[:, 0], main_direction)
        if dot_product < 0:
            eigenvectors[:, 0] = -eigenvectors[:, 0]

    # 将点云转换到主方向坐标系
    transformed_points = np.dot(centered_points, eigenvectors)

    # 计算OBB包围盒的边界
    min_bounds = np.min(transformed_points, axis=0)
    max_bounds = np.max(transformed_points, axis=0)

    # 计算OBB的中心点和尺寸
    obb_center = (min_bounds + max_bounds) / 2
    obb_size = max_bounds - min_bounds

    # 计算内接圆柱体的半径（取垂直于长度方向的两个维度的最小值的一半）
    radius = min(obb_size[1], obb_size[2]) / 2

    # 计算基线的两个端点
    # 长度方向的起点和终点
    length_start = obb_center[0] - obb_size[0] / 2
    length_end = obb_center[0] + obb_size[0] / 2

    # 转换回原始坐标系
    # 起点和终点在主方向坐标系中的坐标
    start_transformed = np.array([length_start, 0, 0])
    end_transformed = np.array([length_end, 0, 0])

    # 转换回原始坐标系
    start_original = centroid + np.dot(start_transformed, eigenvectors.T)
    end_original = centroid + np.dot(end_transformed, eigenvectors.T)

    # 构造SegmentLine
    # 计算线段的方向向量和一点
    segment_direction = end_original - start_original
    segment_direction_normalized = segment_direction / np.linalg.norm(segment_direction)

    # 构造SegmentLine参数 [方向向量, 一点]
    segment_parameters = np.concatenate([segment_direction_normalized, start_original])
    segment_line = SegmentLine(segment_parameters)
    segment_line.link_points = [
        BasePoint(start_original),
        BasePoint(end_original),
    ]

    return segment_line, radius
