# -*- coding: utf-8 -*-
"""
@File    :   noise_utils.py
@Time    :   2025/12/24 22:28:26
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   点云噪声处理工具
"""

import numpy as np


def add_gaussian_noise(point_cloud, mean=0, std=0.01):
    """
    向点云每个点的坐标添加随机高斯分布的偏移量。
    特点：简单高效，适用于模拟激光雷达等传感器的随机误差。

    参数:
    - point_cloud: 点云数据，形状为 (N, 3)，N为点数，3为坐标 (x,y,z)
    - mean: 高斯分布均值
    - std: 高斯分布标准差，控制噪声强度
    """
    noise = np.random.normal(mean, std, size=point_cloud.shape)
    noisy_cloud = point_cloud + noise
    return noisy_cloud


def add_salt_pepper_noise(point_cloud, noise_ratio=0.05, salt_ratio=0.5):
    """
    随机将部分点替换为极大或极小值，模拟传感器异常值。
    特点：产生稀疏的离群点，影响局部点云结构。

    参数:
    - noise_ratio: 受噪声影响的点比例
    - salt_ratio: 椒盐噪声中“盐”（极大值）的比例
    """
    n_points = point_cloud.shape[0]
    n_noise = int(n_points * noise_ratio)
    noise_indices = np.random.choice(n_points, n_noise, replace=False)

    # 随机生成极大值（盐）或极小值（椒）
    for idx in noise_indices:
        if np.random.rand() < salt_ratio:
            # 添加“盐”噪声（极大值）
            point_cloud[idx] += np.random.uniform(0.1, 0.5, 3)
        else:
            # 添加“椒”噪声（极小值）
            point_cloud[idx] -= np.random.uniform(0.1, 0.5, 3)

    return point_cloud


def add_outlier_noise(point_cloud, outlier_ratio=0.1, scale=1.0):
    """
    随机在点云周围生成孤立点，模拟环境干扰或错误检测。
    特点：生成完全独立的噪声点，不依附于原始点云。

    参数:
    - outlier_ratio: 离群点占总点数的比例
    - scale: 离群点分布范围的缩放因子
    """
    n_points = point_cloud.shape[0]

    # 处理空点云情况
    if n_points == 0:
        return point_cloud.copy()

    n_outliers = int(n_points * outlier_ratio)

    # 计算原始点云的边界框
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    range_coords = max_coords - min_coords

    # 在边界框周围随机生成离群点
    outliers = np.random.rand(n_outliers, 3) * range_coords * scale + min_coords

    # 将离群点与原始点云合并
    noisy_cloud = np.vstack([point_cloud, outliers])
    return noisy_cloud


def add_surface_noise(point_cloud, normals, std=0.01):
    """
    沿点云表面的法线方向添加随机偏移，模拟更真实的传感器噪声。
    特点：需要计算点云法线，噪声方向与表面几何相关。

    参数:
    - point_cloud: 点云数据
    - normals: 对应点的法线方向，形状为 (N, 3)
    - std: 噪声强度标准差
    """
    # 确保法线归一化
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # 为每个点生成随机偏移量（沿法线方向）
    offsets = np.random.normal(0, std, size=point_cloud.shape[0])
    noise = normals * offsets[:, np.newaxis]

    noisy_cloud = point_cloud + noise
    return noisy_cloud
