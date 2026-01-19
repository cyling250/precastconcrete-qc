import numpy as np
import random


class RANSACPlaneFitting:
    """
    基于RANSAC算法的点云平面拟合

    参数:
    max_iterations: int, 最大迭代次数
    distance_threshold: float, 点到平面的距离阈值，小于此阈值的点被视为内点
    confidence: float, 置信度，取值范围(0,1)
    """

    def __init__(self, max_iterations=1000, distance_threshold=0.01, confidence=0.99):
        self.max_iterations = max_iterations  # 最大迭代次数
        self.distance_threshold = distance_threshold  # 距离阈值
        self.confidence = confidence  # 置信度

        # 拟合结果
        self.best_plane = None  # [a, b, c, d] 平面方程 ax + by + cz + d = 0
        self.best_inliers = None  # 内点索引
        self.best_inlier_count = 0  # 内点数量

    def fit(self, points):
        """
        对输入的点云进行平面拟合

        参数:
        points: numpy.ndarray, 形状为 (n_points, 3) 的点云数据

        返回:
        best_plane: numpy.ndarray, 最佳拟合平面参数 [a, b, c, d]
        best_inliers: numpy.ndarray, 内点索引
        """
        n_points = points.shape[0]

        # 检查点云数量是否足够
        if n_points < 3:
            raise ValueError("点云数量必须至少为3个")

        # 计算所需的最小迭代次数
        required_iterations = self._calculate_required_iterations(n_points)
        actual_iterations = min(required_iterations, self.max_iterations)

        for i in range(actual_iterations):
            # 随机选择3个点
            sample_indices = random.sample(range(n_points), 3)
            sample_points = points[sample_indices]

            # 拟合平面
            plane = self._fit_plane(sample_points)

            # 计算所有点到平面的距离
            distances = self._compute_distances(points, plane)

            # 找出内点
            inliers = np.where(distances < self.distance_threshold)[0]
            inlier_count = len(inliers)

            # 更新最佳拟合平面
            if inlier_count > self.best_inlier_count:
                self.best_inlier_count = inlier_count
                self.best_plane = plane
                self.best_inliers = inliers

                # 更新所需的迭代次数
                required_iterations = self._calculate_required_iterations(n_points)
                actual_iterations = min(required_iterations, self.max_iterations)

        # 对最佳内点重新拟合平面，提高精度
        if self.best_inliers is not None and len(self.best_inliers) > 3:
            inlier_points = points[self.best_inliers]
            self.best_plane = self._fit_plane(inlier_points)

        return self.best_plane, self.best_inliers

    def _calculate_required_iterations(self, n_points):
        """
        计算所需的最小迭代次数

        参数:
        n_points: int, 点云数量

        返回:
        int, 所需的迭代次数
        """
        if self.best_inlier_count == 0:
            inlier_ratio = 0.1  # 默认内点比例
        else:
            inlier_ratio = self.best_inlier_count / n_points

        if inlier_ratio == 0:
            return self.max_iterations

        # 计算所需的迭代次数
        numerator = np.log(1 - self.confidence)
        denominator = np.log(1 - np.power(inlier_ratio, 3))

        if denominator == 0:
            return self.max_iterations

        return int(numerator / denominator) + 1

    def _fit_plane(self, points):
        """
        从3个点拟合平面

        参数:
        points: numpy.ndarray, 形状为 (3, 3) 的3个点

        返回:
        numpy.ndarray, 平面参数 [a, b, c, d]
        """
        # 计算两个向量
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]

        # 计算法向量
        normal = np.cross(v1, v2)

        # 确保法向量不为零
        if np.linalg.norm(normal) < 1e-10:
            return None

        # 归一化法向量
        normal = normal / np.linalg.norm(normal)

        # 计算d参数
        d = -np.dot(normal, points[0])

        return np.array([normal[0], normal[1], normal[2], d])

    def _compute_distances(self, points, plane):
        """
        计算所有点到平面的距离

        参数:
        points: numpy.ndarray, 点云数据
        plane: numpy.ndarray, 平面参数 [a, b, c, d]

        返回:
        numpy.ndarray, 每个点到平面的距离
        """
        a, b, c, d = plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        return distances

    def get_plane_equation(self):
        """
        获取平面方程的字符串表示

        返回:
        str, 平面方程
        """
        if self.best_plane is None:
            return "未拟合平面"

        a, b, c, d = self.best_plane
        return f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0"

    def get_normal_vector(self):
        """
        获取平面的法向量

        返回:
        numpy.ndarray, 法向量 [a, b, c]
        """
        if self.best_plane is None:
            return None

        return self.best_plane[:3]

    def get_inlier_points(self, points):
        """
        获取内点

        参数:
        points: numpy.ndarray, 原始点云数据

        返回:
        numpy.ndarray, 内点
        """
        if self.best_inliers is None:
            return None

        return points[self.best_inliers]

    def get_outlier_points(self, points):
        """
        获取外点

        参数:
        points: numpy.ndarray, 原始点云数据

        返回:
        numpy.ndarray, 外点
        """
        if self.best_inliers is None:
            return None

        all_indices = np.arange(points.shape[0])
        outlier_indices = np.setdiff1d(all_indices, self.best_inliers)
        return points[outlier_indices]
