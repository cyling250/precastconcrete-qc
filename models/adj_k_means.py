import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize


class AdjustKMeans:
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

    def normal(self, X):
        """
        对输入的点云特征向量进行归一化

        参数:
        X: numpy.ndarray, 输入的点云特征向量

        返回:
        X_normalized: numpy.ndarray, 归一化后的点云特征向量
        """
        # TODO 因为有距离的限制，需要对特征向量进行归一化
        pass

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
