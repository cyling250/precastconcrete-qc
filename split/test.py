# -*- coding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2026/01/15 11:34:13
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   从npy文件加载点云特征并依次可视化
"""

import open3d as o3d
import numpy as np
import os


def visualize_features_from_npy(
    point_cloud_file, features_npy_file, distance, feature_name
):
    """
    从npy文件加载特征数据并可视化

    参数:
    point_cloud_file: str, 原始点云文件路径
    features_npy_file: str, 特征数据npy文件路径
    distance: int, 邻域距离
    feature_name: str, 特征名称
    """
    # 读取原始点云
    pcd = o3d.io.read_point_cloud(point_cloud_file)
    points = np.asarray(pcd.points)

    # 从npy文件加载特征数据
    features_dict = np.load(features_npy_file, allow_pickle=True).item()

    # 检查距离和特征名称是否存在
    if distance not in features_dict:
        print(f"错误：距离 {distance}mm 不存在于特征数据中")
        return

    if feature_name not in features_dict[distance]:
        print(f"错误：特征 {feature_name} 不存在于距离 {distance}mm 的数据中")
        return

    # 获取特征数据
    feature_values = features_dict[distance][feature_name]

    # 归一化特征值到 [0, 1] 范围
    min_val = np.min(feature_values)
    max_val = np.max(feature_values)
    if max_val - min_val == 0:
        normalized = np.zeros_like(feature_values)
    else:
        normalized = (feature_values - min_val) / (max_val - min_val)

    # 创建可视化点云
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = pcd.points

    # 设置灰度颜色
    colors = np.zeros((len(feature_values), 3))
    colors[:, 0] = normalized  # R
    colors[:, 1] = normalized  # G
    colors[:, 2] = normalized  # B
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)

    # 设置窗口标题
    window_title = f"{distance}-{feature_name}"

    # 可视化点云
    print(f"正在可视化距离 {distance}mm 的{feature_name}特征...")
    o3d.visualization.draw_geometries([vis_pcd], window_name=window_title)


if __name__ == "__main__":
    # 原始点云文件路径
    point_cloud_file = "mock/res/point/预制叠合梁300x600.ply"

    # 特征数据npy文件路径
    features_npy_file = "output/neighborhood_features/all_features.npy"

    # 检查文件是否存在
    if not os.path.exists(point_cloud_file):
        print(f"错误：原始点云文件不存在：{point_cloud_file}")
        exit(1)

    if not os.path.exists(features_npy_file):
        print(f"错误：特征数据文件不存在：{features_npy_file}")
        print("请先运行 point_neighborhood_features.py 生成特征数据")
        exit(1)

    # 从npy文件加载特征数据，获取可用的距离和特征名称
    features_dict = np.load(features_npy_file, allow_pickle=True).item()
    distances = list(features_dict.keys())
    feature_names = list(features_dict[distances[0]].keys()) if distances else []

    print(f"可用的距离：{distances}")
    print(f"可用的特征：{feature_names}")

    # 依次可视化所有距离的所有特征
    # 这里我们只可视化部分距离和特征作为示例
    # 可以根据需要修改要可视化的距离和特征
    distances_to_visualize = [50, 100]  # [10, 20, 30, 40, 50, 60, 70, 80, 90]
    feature_to_visualize = "linearity"

    for distance in distances_to_visualize:
        if distance in distances and feature_to_visualize in feature_names:
            visualize_features_from_npy(
                point_cloud_file, features_npy_file, distance, feature_to_visualize
            )
