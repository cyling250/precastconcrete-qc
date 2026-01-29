# -*- coding: utf-8 -*-
"""
@File    :   point_neighborhood_features.py
@Time    :   2026/01/15 15:00:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   计算点云中每个点在不同距离邻域内的特征，并保存为单个npy文件
"""

import os
import numpy as np
from scipy.spatial import KDTree
from utils.eigen_measures_utils import EigenMeasures
from tqdm import tqdm
import open3d as o3d

DEFAULT_NEIGHBOR_FEATURES = [
    "linearity",
    "planarity",
    "sphericity",
    "omnivariance",
    "anisotropy",
    "eigenentropy",
    "tortuosity",
    "curvature_change",
]


def neighbor_features(
    point_cloud: np.ndarray,
    output_file: str,
    distance: float,
    feature_names=DEFAULT_NEIGHBOR_FEATURES,
    is_visualize: bool = False,
):
    """
    计算点云中每个点在不同距离邻域内的特征

    参数:
    point_cloud: np.ndarray, 形状为 (n_points, 3) 的点云数组
    output_file: str, 输出npy文件路径
    distance: list, 邻域距离列表（单位：mm）
    feature_names: list, 要计算的特征名称列表（默认：DEFAULT_NEIGHBOR_FEATURES）

    返回:
    dict, 包含每个距离下的特征字典
    """
    # 校验参数
    assert isinstance(point_cloud, np.ndarray), "point_cloud 必须是 numpy.ndarray 类型"
    assert point_cloud.shape[1] == 3, "point_cloud 必须是 (n_points, 3) 形状"

    # 构建KD树以加速邻域搜索
    kdtree = KDTree(point_cloud)
    # 初始化特征数组
    features = {
        "X": np.zeros(point_cloud.shape[0]),
        "Y": np.zeros(point_cloud.shape[0]),
        "Z": np.zeros(point_cloud.shape[0]),
    }
    features.update({name: np.zeros(point_cloud.shape[0]) for name in feature_names})
    # 为计算XYZ准备参数
    centroid = np.mean(point_cloud, axis=0)
    point_cloud_c = point_cloud - centroid
    m = np.max(np.sqrt(np.sum(point_cloud_c**2, axis=1)))
    point_cloud_c = point_cloud_c / m
    # 对于每个点，计算其邻域特征
    for i in tqdm(range(point_cloud.shape[0]), desc=f"计算距离 {distance}mm 邻域特征"):
        # 找到邻域内的点
        indices = kdtree.query_ball_point(point_cloud[i], distance / 1000.0)  # 转换为米
        # 如果邻域内点太少，跳过
        if len(indices) < 3:
            continue
        # 获取邻域点
        neighborhood = point_cloud[indices]
        # 计算特征
        eigen_measures = EigenMeasures(neighborhood)
        # 保存特征
        features["X"][i] = point_cloud_c[i, 0]
        features["Y"][i] = point_cloud_c[i, 1]
        features["Z"][i] = point_cloud_c[i, 2]
        for name in feature_names:
            features[name][i] = getattr(eigen_measures, name)
    if is_visualize:
        visualize_features(point_cloud, features, feature_names)
    # 保存所有特征到单个npy文件
    save_features(features, output_file)


def visualize_features(point_cloud, features, feature_names):
    # DEBUG 可视化邻域特征
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    for name in feature_names:
        # 归一化特征值到[0, 1]范围
        features[name] = (features[name] - np.min(features[name])) / (
            np.max(features[name]) - np.min(features[name])
        )
        # 转换为RGB颜色（0-1范围）
        colors = np.zeros_like(point_cloud)
        colors[:, 0] = features[name]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])


def save_features(features_dict: dict, output_file: str):
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
