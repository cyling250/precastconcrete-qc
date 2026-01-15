# -*- coding: utf-8 -*-
"""
@File    :   k_means.py
@Time    :   2026/01/15 16:00:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   点云k-means聚类算法，基于sklearn实现，支持自定义距离计算函数
"""

import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize


class PointCloudKMeans:
    """
    点云k-means聚类算法，基于sklearn实现，支持自定义距离计算函数

    参数:
    n_clusters: int, 聚类数量
    max_iter: int, 最大迭代次数
    distance_metric: str, 距离度量方式，可选：'euclidean'（欧氏距离）、'manhattan'（曼哈顿距离）、'cosine'（余弦距离）
    tol: float, 收敛阈值
    random_state: int, 随机种子
    """

    def __init__(
        self,
        n_clusters=8,
        max_iter=100,
        distance_metric="euclidean",
        tol=1e-4,
        random_state=None,
        feature_weights=None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.tol = tol
        self.random_state = random_state
        self.feature_weights = feature_weights
        self.kmeans = None
        self.labels_ = None
        self.centroids = None
        self.inertia_ = None

    def fit(self, X):
        """
        对输入的点云特征向量进行聚类

        参数:
        X: numpy.ndarray, 形状为 (n_samples, n_features)，包含n_samples个点的n_features维特征向量

        返回:
        self: 聚类模型
        """
        start_time = time.time()

        # 应用特征权重
        if self.feature_weights is not None:
            # 确保权重数量与特征数量匹配
            if len(self.feature_weights) != X.shape[1]:
                raise ValueError(
                    f"特征权重数量 ({len(self.feature_weights)}) 与特征数量 ({X.shape[1]}) 不匹配"
                )

            # 将权重转换为数组
            weights = np.array(self.feature_weights)

            # 对特征向量应用权重
            X_weighted = X * weights
            print(f"已应用特征权重: {self.feature_weights}")
        else:
            X_weighted = X
            weights = None

        # 根据距离度量方式选择处理方法
        if self.distance_metric == "euclidean":
            # 欧氏距离直接使用sklearn的KMeans
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            self.kmeans.fit(X_weighted)

            # 保存结果
            self.labels_ = self.kmeans.labels_
            self.centroids = (
                self.kmeans.cluster_centers_ / weights
                if weights is not None
                else self.kmeans.cluster_centers_
            )
            self.inertia_ = self.kmeans.inertia_

        elif self.distance_metric == "manhattan":
            # 曼哈顿距离使用自定义实现
            self._custom_kmeans(X, weights, distance_metric="manhattan")

        elif self.distance_metric == "cosine":
            # 余弦距离需要先归一化，权重应用方式不同
            if weights is not None:
                print("警告：余弦距离下特征权重的效果可能不如欧氏距离明显")
                X_weighted = X * weights
            X_normalized = normalize(X_weighted, axis=1)
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            self.kmeans.fit(X_normalized)

            # 保存结果
            self.labels_ = self.kmeans.labels_
            self.centroids = (
                self.kmeans.cluster_centers_ / weights
                if weights is not None
                else self.kmeans.cluster_centers_
            )
            self.inertia_ = self.kmeans.inertia_

        else:
            raise ValueError(f"不支持的距离度量方式: {self.distance_metric}")

        end_time = time.time()
        print(f"聚类完成，耗时: {end_time - start_time:.2f}秒")
        return self

    def _custom_kmeans(self, X, weights, distance_metric="manhattan"):
        """
        自定义k-means实现，支持曼哈顿距离和特征权重

        参数:
        X: numpy.ndarray, 输入的点云特征向量
        weights: numpy.ndarray, 特征权重
        distance_metric: str, 距离度量方式
        """
        n_samples, n_features = X.shape

        # 初始化质心
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()

        # 初始化标签
        self.labels_ = np.zeros(n_samples, dtype=int)

        # 开始迭代
        for i in range(self.max_iter):
            # 保存当前质心
            old_centroids = self.centroids.copy()

            # 分配每个点到最近的质心
            self._assign_labels(X, weights, distance_metric)

            # 更新质心
            self._update_centroids(X)

            # 计算质心变化量
            centroid_shift = np.mean(
                np.linalg.norm(self.centroids - old_centroids, axis=1)
            )

            # 检查是否收敛
            if centroid_shift < self.tol:
                print(f"自定义k-means算法在第 {i + 1} 次迭代时收敛")
                break

        # 计算惯性
        self._compute_inertia(X, weights, distance_metric)

    def _assign_labels(self, X, weights, distance_metric):
        """
        分配每个点到最近的质心

        参数:
        X: numpy.ndarray, 输入的点云特征向量
        weights: numpy.ndarray, 特征权重
        distance_metric: str, 距离度量方式
        """
        if weights is not None:
            # 对特征向量和质心应用权重
            X_weighted = X * weights
            centroids_weighted = self.centroids * weights

            # 计算所有点到所有质心的距离
            distances = pairwise_distances(
                X_weighted, centroids_weighted, metric=distance_metric
            )
        else:
            # 计算所有点到所有质心的距离
            distances = pairwise_distances(X, self.centroids, metric=distance_metric)

        # 分配每个点到距离最小的质心
        self.labels_ = np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        """
        更新质心

        参数:
        X: numpy.ndarray, 输入的点云特征向量
        """
        for cluster_idx in range(self.n_clusters):
            # 获取该聚类中的所有点
            cluster_points = X[self.labels_ == cluster_idx]

            # 如果聚类中没有点，保留原质心
            if len(cluster_points) == 0:
                continue

            # 更新质心为该聚类中所有点的平均值
            self.centroids[cluster_idx] = np.mean(cluster_points, axis=0)

    def _compute_inertia(self, X, weights, distance_metric):
        """
        计算惯性（所有点到其最近质心的距离平方和）

        参数:
        X: numpy.ndarray, 输入的点云特征向量
        weights: numpy.ndarray, 特征权重
        distance_metric: str, 距离度量方式
        """
        inertia = 0
        for i, point in enumerate(X):
            centroid = self.centroids[self.labels_[i]]

            if weights is not None:
                # 应用权重
                point_weighted = point * weights
                centroid_weighted = centroid * weights
            else:
                point_weighted = point
                centroid_weighted = centroid

            if distance_metric == "manhattan":
                distance = np.sum(np.abs(point_weighted - centroid_weighted))
            else:
                distance = np.linalg.norm(point_weighted - centroid_weighted)
            inertia += distance**2
        self.inertia_ = inertia

    def predict(self, X):
        """
        预测新的点云特征向量所属的聚类

        参数:
        X: numpy.ndarray, 形状为 (n_samples, n_features)，新的点云特征向量

        返回:
        labels: numpy.ndarray, 形状为 (n_samples,)，每个点所属的聚类标签
        """
        if self.feature_weights is not None:
            weights = np.array(self.feature_weights)
            X_weighted = X * weights
        else:
            X_weighted = X
            weights = None

        if self.distance_metric == "euclidean":
            return self.kmeans.predict(X_weighted)
        elif self.distance_metric == "cosine":
            X_normalized = normalize(X_weighted, axis=1)
            return self.kmeans.predict(X_normalized)
        else:
            # 使用自定义预测实现
            if self.distance_metric == "manhattan":
                # 计算所有点到所有质心的距离
                if weights is not None:
                    centroids_weighted = self.centroids * weights
                    distances = pairwise_distances(
                        X_weighted, centroids_weighted, metric="manhattan"
                    )
                else:
                    distances = pairwise_distances(
                        X, self.centroids, metric="manhattan"
                    )
                return np.argmin(distances, axis=1)
            else:
                raise ValueError(f"不支持的距离度量方式: {self.distance_metric}")


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
        kmeans = PointCloudKMeans(
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
