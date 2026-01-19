# -*- coding: utf-8 -*-
"""
@File    :   k_means.py
@Time    :   2026/01/15 16:00:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   点云k-means聚类算法，基于sklearn实现，支持自定义距离计算函数
"""

import numpy as np
import open3d as o3d
from models.adj_k_means import AdjustKMeans


def create_feature_vector(features_npy_file: str):
    """
    创建点云特征向量，将空间坐标和局部几何特征组合

    参数:
    features_npy_file: str, 特征数据npy文件路径

    返回:
    feature_vectors: numpy.ndarray, 形状为 (n_points, n_features)，点云特征向量
    feature_names: list, 特征名称列表
    """

    # 从npy文件加载特征数据
    features_dict = np.load(features_npy_file, allow_pickle=True).item()

    # 获取所有特征名称
    feature_names = list(features_dict.keys())

    # 创建特征向量
    feature_vectors_list = []

    # 添加所有特征
    for feature_name in features_dict.keys():
        feature_values = features_dict[feature_name]
        # 确保特征值形状正确
        feature_values = feature_values.reshape(-1, 1)
        feature_vectors_list.append(feature_values)

    # 组合所有特征
    feature_vectors = np.hstack(feature_vectors_list)

    return feature_vectors, feature_names


def save_clusters(
    point_cloud: o3d.geometry.PointCloud, labels: np.ndarray, output_file=None
):
    """
    保存聚类结果到点云文件

    参数:
    point_cloud: numpy.ndarray, 形状为 (n_points, 3)，点云数据
    labels: numpy.ndarray, 每个点的聚类标签
    output_file: str, 输出文件路径（可选）
    """
    # 为每个聚类生成随机颜色
    n_clusters = len(np.unique(labels))
    colors = np.random.rand(n_clusters, 3)

    # 为每个点分配颜色
    point_colors = colors[labels]

    # 设置点云颜色
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)

    # 保存结果
    if output_file:
        o3d.io.write_point_cloud(output_file, point_cloud)


def clustering(
    point_cloud: o3d.geometry.PointCloud,
    features_npy_file: str,
    output_file: str,
    feature_weights: list,
    n_clusters=2,  # 2类：钢筋和混凝土
    max_iter=100,
    distance_metric="euclidean",  # 只使用欧氏距离，因为它最适合特征权重
):
    """
    对点云进行K-Means聚类分析，区分钢筋和混凝土

    参数:
    point_cloud: o3d.geometry.PointCloud, 输入点云
    features_npy_file: str, 特征数据npy文件路径
    output_file: str, 输出聚类结果文件路径（.ply格式）
    feature_weights: list, 特征权重列表，默认：DEFAULT_FEATURE_WEIGHTS
    n_clusters: int, 聚类簇数，默认：2
    max_iter: int, 最大迭代次数，默认：100
    distance_metric: str, 距离度量方式，默认："euclidean"
    distance: int, 邻域距离（mm），默认：50

    返回:
    None
    """
    # 创建点云特征向量
    feature_vectors, _ = create_feature_vector(
        features_npy_file,
    )

    # 创建并训练聚类模型
    kmeans = AdjustKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        distance_metric=distance_metric,
        tol=1e-4,
        random_state=42,
        feature_weights=feature_weights,
    )

    kmeans.fit(feature_vectors)
    # 保存聚类结果
    save_clusters(
        point_cloud,
        kmeans.labels_,
        output_file,
    )
    print(f"聚类结果已保存到 {output_file}")
