# -*- coding: utf-8 -*-
"""
@File    :   k_means.py
@Time    :   2026/01/15 16:00:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   点云k-means聚类算法，基于sklearn实现，支持自定义距离计算函数
"""

import numpy as np
from models.adj_k_means import AdjustKMeans

def create_point_cloud_feature_vector(
    point_cloud_file, features_npy_file, distance=10, include_coordinates=True
):
    """
    创建点云特征向量，将空间坐标和局部几何特征组合

    参数:
    point_cloud_file: str, 原始点云文件路径
    features_npy_file: str, 特征数据npy文件路径
    distance: int, 邻域距离（mm）
    include_coordinates: bool, 是否包含空间坐标

    返回:
    feature_vectors: numpy.ndarray, 形状为 (n_points, n_features)，点云特征向量
    feature_names: list, 特征名称列表
    """
    import open3d as o3d
    import numpy as np

    # 读取原始点云
    pcd = o3d.io.read_point_cloud(point_cloud_file)
    points = np.asarray(pcd.points)

    # 从npy文件加载特征数据
    features_dict = np.load(features_npy_file, allow_pickle=True).item()

    # 检查距离是否存在
    if distance not in features_dict:
        raise ValueError(f"错误：距离 {distance}mm 不存在于特征数据中")

    # 获取所有特征名称
    feature_names = list(features_dict[distance].keys())

    # 创建特征向量
    feature_vectors_list = []

    # 如果包含空间坐标，先添加坐标
    if include_coordinates:
        feature_vectors_list.append(points)
        feature_names = ["X", "Y", "Z"] + feature_names

    # 添加所有特征
    for feature_name in features_dict[distance].keys():
        feature_values = features_dict[distance][feature_name]
        # 确保特征值形状正确
        feature_values = feature_values.reshape(-1, 1)
        feature_vectors_list.append(feature_values)

    # 组合所有特征
    feature_vectors = np.hstack(feature_vectors_list)

    return feature_vectors, feature_names


def visualize_clusters(point_cloud_file, labels, output_file=None):
    """
    可视化聚类结果

    参数:
    point_cloud_file: str, 原始点云文件路径
    labels: numpy.ndarray, 每个点的聚类标签
    output_file: str, 输出文件路径（可选）
    """
    import open3d as o3d
    import numpy as np

    # 读取原始点云
    pcd = o3d.io.read_point_cloud(point_cloud_file)

    # 为每个聚类生成随机颜色
    n_clusters = len(np.unique(labels))
    colors = np.random.rand(n_clusters, 3)

    # 为每个点分配颜色
    point_colors = colors[labels]

    # 设置点云颜色
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # 保存结果
    if output_file:
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"聚类结果已保存到 {output_file}")

    # 可视化
    print("正在可视化聚类结果...")
    o3d.visualization.draw_geometries([pcd], window_name="点云聚类结果")


def main():
    """
    示例用法：将点云聚类为钢筋和混凝土两类
    """
    print("开始点云k-means聚类示例...")

    # 定义文件路径
    point_cloud_file = "mock/res/point/预制叠合梁300x600.ply"
    features_npy_file = "output/neighborhood_features/all_features.npy"

    # 选择邻域距离
    distance = 50  # 50mm

    # 创建点云特征向量
    print(f"正在创建距离 {distance}mm 下的点云特征向量...")
    feature_vectors, feature_names = create_point_cloud_feature_vector(
        point_cloud_file, features_npy_file, distance=distance, include_coordinates=True
    )

    print(f"特征向量形状: {feature_vectors.shape}")
    print(f"特征名称: {feature_names}")

    # 设置特征权重
    # 根据需求：X、Y、Z坐标权重最低，线性度和平面度权重最高
    # 特征名称顺序：['X', 'Y', 'Z', 'linearity', 'planarity', 'sphericity', 'omnivariance', 'anisotropy', 'eigenentropy', 'tortuosity', 'curvature_change']
    feature_weights = [
        0.5,  # X坐标
        0.5,  # Y坐标
        0.5,  # Z坐标
        3.0,  # linearity（线性度）
        3.0,  # planarity（平面度）
        1.0,  # sphericity（球度）
        1.0,  # omnivariance（各向同性）
        1.0,  # anisotropy（各向异性）
        1.0,  # eigenentropy（特征熵）
        1.0,  # tortuosity（曲折度）
        1.0,  # curvature_change（曲率变化）
    ]

    print(f"\n设置的特征权重: {feature_weights}")
    print(f"对应的特征名称: {feature_names}")

    # 设置聚类参数
    n_clusters = 2  # 2类：钢筋和混凝土
    max_iter = 100
    distance_metrics = ["euclidean"]  # 只使用欧氏距离，因为它最适合特征权重

    # 对每种距离度量方式进行聚类
    for distance_metric in distance_metrics:
        print(
            f"\n使用 {distance_metric} 距离度量进行聚类，分为 {n_clusters} 类（钢筋和混凝土）..."
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

        # 打印聚类结果
        print(f"惯性: {kmeans.inertia_:.2f}")
        print(f"聚类中心: {kmeans.centroids.shape}")

        # 统计每个聚类的点数量
        cluster_counts = np.bincount(kmeans.labels_)
        for cluster_idx, count in enumerate(cluster_counts):
            material = "钢筋" if cluster_idx == 0 else "混凝土"
            print(f"聚类 {cluster_idx} ({material}): {count} 个点")

        # 可视化聚类结果（只显示一次，使用欧氏距离的结果）
        if distance_metric == "euclidean":
            output_file = f"output/cluster_results_{n_clusters}clusters_{distance}mm_{distance_metric}.ply"
            visualize_clusters(point_cloud_file, kmeans.labels_, output_file)


if __name__ == "__main__":
    main()
