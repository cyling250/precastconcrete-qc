# -*- coding: utf-8 -*-
"""
@File    :   add_noise.py
@Time    :   2025/12/24 22:28:26
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   向点云添加噪声
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


if __name__ == "__main__":
    import open3d as o3d

    # 使用./mock/res/pcd/wall_and_beam_1000000p.pcd文件测试所有函数，并可视化结果
    file_path = "./mock/res/pcd/wall_and_beam_1000000p.pcd"

    print(f"加载点云文件: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.5)  # 体素降采样
    point_cloud = np.asarray(pcd_down.points)

    # 为表面噪声准备法线（垂直于Y轴的法线）
    normals = np.tile([0, 1, 0], (point_cloud.shape[0], 1))

    # 逐步应用噪声并逐个可视化
    print("\n开始逐步测试噪声函数...")

    # 1. 可视化原始点云
    print("\n1. 显示原始点云")
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(point_cloud)
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
    o3d.visualization.draw_geometries(
        [original_pcd],
        window_name="1. 原始点云",
        width=1200,
        height=800,
        left=50,
        top=50,
    )

    # 2. 添加高斯噪声并可视化
    print("\n2. 添加高斯噪声")
    noisy_cloud_gaussian = add_gaussian_noise(point_cloud.copy(), std=0.05)
    gaussian_pcd = o3d.geometry.PointCloud()
    gaussian_pcd.points = o3d.utility.Vector3dVector(noisy_cloud_gaussian)
    gaussian_pcd.paint_uniform_color([1, 0, 0])  # 红色
    o3d.visualization.draw_geometries(
        [gaussian_pcd],
        window_name="2. 原始点云 + 高斯噪声",
        width=1200,
        height=800,
        left=50,
        top=50,
    )

    # 3. 在高斯噪声基础上添加椒盐噪声并可视化
    print("\n3. 添加椒盐噪声")
    noisy_cloud_sp = add_salt_pepper_noise(noisy_cloud_gaussian.copy(), noise_ratio=0.1)
    sp_pcd = o3d.geometry.PointCloud()
    sp_pcd.points = o3d.utility.Vector3dVector(noisy_cloud_sp)
    sp_pcd.paint_uniform_color([0, 1, 0])  # 绿色
    o3d.visualization.draw_geometries(
        [sp_pcd],
        window_name="3. 原始点云 + 高斯噪声 + 椒盐噪声",
        width=1200,
        height=800,
        left=50,
        top=50,
    )

    # 4. 在高斯+椒盐噪声基础上添加离群点噪声并可视化
    print("\n4. 添加离群点噪声")
    noisy_cloud_outlier = add_outlier_noise(noisy_cloud_sp.copy(), outlier_ratio=0.2)
    outlier_pcd = o3d.geometry.PointCloud()
    outlier_pcd.points = o3d.utility.Vector3dVector(noisy_cloud_outlier)
    outlier_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
    o3d.visualization.draw_geometries(
        [outlier_pcd],
        window_name="4. 原始点云 + 高斯噪声 + 椒盐噪声 + 离群点噪声",
        width=1200,
        height=800,
        left=50,
        top=50,
    )

    # 5. 在所有噪声基础上添加表面噪声并可视化
    print("\n5. 添加表面噪声")
    # 注意：离群点噪声增加了点的数量，需要调整法线数量
    normals_extended = np.vstack(
        [
            normals,
            np.tile(
                [0, 1, 0], (noisy_cloud_outlier.shape[0] - point_cloud.shape[0], 1)
            ),
        ]
    )
    noisy_cloud_surface = add_surface_noise(
        noisy_cloud_outlier.copy(), normals_extended, std=0.02
    )
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(noisy_cloud_surface)
    surface_pcd.paint_uniform_color([1, 1, 0])  # 黄色
    o3d.visualization.draw_geometries(
        [surface_pcd],
        window_name="5. 原始点云 + 所有噪声",
        width=1200,
        height=800,
        left=50,
        top=50,
    )

    print("\n测试完成！")
