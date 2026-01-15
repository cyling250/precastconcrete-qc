# -*- coding: utf-8 -*-
"""
@File    :   point_neighborhood_features.py
@Time    :   2026/01/15 15:00:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   计算点云中每个点在不同距离邻域内的特征，并保存为单个npy文件
"""

import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# 添加项目根目录到Python路径，确保能找到utils模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eigen_measures_utils import EigenMeasures


def calculate_neighborhood_features(pcd_file, distances=[1, 10, 100, 200]):
    """
    计算点云中每个点在不同距离邻域内的特征

    参数:
    pcd_file: str, 点云文件路径
    distances: list, 邻域距离列表（单位：mm）

    返回:
    dict, 包含每个距离下的特征字典
    """
    # 读取点云文件
    print(f"正在读取点云文件: {pcd_file}")
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    print(f"点云包含 {points.shape[0]} 个点")

    # 构建KD树以加速邻域搜索
    print("正在构建KD树...")
    kdtree = KDTree(points)

    # 定义要计算的特征名称
    feature_names = [
        "linearity",
        "planarity",
        "sphericity",
        "omnivariance",
        "anisotropy",
        "eigenentropy",
        "tortuosity",
        "curvature_change",
    ]

    # 创建结果字典
    results = {}

    # 对于每个距离，计算所有点的邻域特征
    for distance in distances:
        print(f"\n正在计算距离 {distance}mm 下的邻域特征...")

        # 初始化特征数组
        features = {name: np.zeros(points.shape[0]) for name in feature_names}

        # 对于每个点，计算其邻域特征
        for i in range(points.shape[0]):
            if i % 1000 == 0:
                print(f"处理点 {i}/{points.shape[0]}")

            # 找到邻域内的点
            indices = kdtree.query_ball_point(points[i], distance / 1000.0)  # 转换为米

            # 如果邻域内点太少，跳过
            if len(indices) < 3:
                continue

            # 获取邻域点
            neighborhood = points[indices]

            # 计算特征
            eigen_measures = EigenMeasures(neighborhood)

            # 保存特征
            for name in feature_names:
                features[name][i] = getattr(eigen_measures, name)

        # 保存结果
        results[distance] = features

    return results


def save_features_to_single_file(features_dict, output_file):
    """
    将所有特征保存到单个npy文件中，使用dict格式存储

    参数:
    features_dict: dict, 包含每个距离下的特征字典
    output_file: str, 输出npy文件路径
    """
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存所有特征到单个npy文件
    np.save(output_file, features_dict)
    print(f"已保存所有特征到 {output_file}")


def main():
    """
    主函数
    """
    # 点云文件路径
    pcd_file = "mock/res/point/预制叠合梁300x600.ply"

    # 输出文件
    output_file = "output/neighborhood_features/all_features.npy"

    # 邻域距离列表（单位：mm）
    distances = [50, 100]

    # 计算邻域特征
    features_dict = calculate_neighborhood_features(pcd_file, distances)

    # 保存特征到单个文件
    save_features_to_single_file(features_dict, output_file)

    print("\n所有操作完成！")


if __name__ == "__main__":
    main()
