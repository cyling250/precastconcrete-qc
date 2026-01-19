# -*- coding: utf-8 -*-
"""
@File    :   eigen_measures_utils.py
@Time    :   2026/01/15 11:52:49
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   计算与点云的特征值和特征向量相关的参数的工具类
"""

import numpy as np
import open3d as o3d


class EigenMeasures:
    """
    点云特征值和特征向量相关参数的计算工具类

    参数:
    cloudData: 点云数据，可以是以下格式：
        - open3d.geometry.PointCloud 对象
        - numpy.ndarray 数组，形状为 (N, 3)
        - 字符串，点云文件路径（.ply, .pcd等格式）
    """

    def __init__(self, cloudData):
        """初始化工具类，直接计算特征值相关参数，不保存点云数据"""
        # 直接处理点云数据，不保存原始点云
        points = self._load_point_cloud(cloudData)

        # 计算特征值和特征向量
        self._eigenvalues, self._eigenvectors = self._calculate_eigen(points)

        # 计算各种特征指标
        self._calculate_measures()

    def _load_point_cloud(self, cloudData):
        """加载点云数据，支持多种输入格式"""
        if isinstance(cloudData, o3d.geometry.PointCloud):
            points = np.asarray(cloudData.points)
        elif isinstance(cloudData, np.ndarray):
            if cloudData.shape[1] != 3:
                raise ValueError("点云数组必须是 (N, 3) 形状")
            points = cloudData.copy()  # 复制数据以避免外部修改影响
        elif isinstance(cloudData, str):
            pcd = o3d.io.read_point_cloud(cloudData)
            points = np.asarray(pcd.points)
        else:
            raise TypeError(
                "cloudData 必须是 open3d.PointCloud、numpy.ndarray 或文件路径字符串"
            )

        return points

    def _calculate_eigen(self, points):
        """计算协方差矩阵、特征值和特征向量"""
        if points is None or len(points) < 3:
            raise ValueError("点云数据不足，无法计算特征值")

        # 计算协方差矩阵
        cov_matrix = np.cov(points.T)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 如果协方差矩阵数值不稳定，尝试添加微小的正则化
        if np.isnan(eigenvalues).any() or np.isinf(eigenvalues).any():
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 排序特征值和特征向量（从大到小）
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        return eigenvalues, eigenvectors

    def _calculate_measures(self):
        """计算各种特征指标并保存为实例属性"""
        eigenvalues = self._eigenvalues

        # 归一化特征值
        e_sum = eigenvalues.sum()
        self.e1, self.e2, self.e3 = (
            eigenvalues / e_sum if e_sum > 0 else [1 / 3, 1 / 3, 1 / 3]
        )

        # 计算各种特征并直接保存为实例属性
        self.linearity = (
            (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
            if eigenvalues[0] != 0
            else 0
        )  # 线性度
        self.planarity = (
            (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
            if eigenvalues[0] != 0
            else 0
        )  # 平面度
        self.sphericity = (
            eigenvalues[2] / eigenvalues[0] if eigenvalues[0] != 0 else 0
        )  # 球形度
        self.omnivariance = np.cbrt(
            eigenvalues[0] * eigenvalues[1] * eigenvalues[2]
        )  # 全方差
        self.anisotropy = (
            (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]
            if eigenvalues[0] != 0
            else 0
        )  # 各向异性
        self.eigenentropy = -np.sum(
            [e * np.log(e) for e in [self.e1, self.e2, self.e3] if e > 0]
        )  # 特征熵
        self.tortuosity = (2 / np.pi) * np.arctan(eigenvalues.sum())  # 凹型度
        self.curvature_change = (
            3 * eigenvalues[2] / e_sum if e_sum > 0 else 0
        )  # 曲率变化

    def print_results(self):
        """打印计算结果"""
        print(
            f"Normalized eigenvalues: e1={self.e1:.6f}, e2={self.e2:.6f}, e3={self.e3:.6f}"
        )
        print(f"Linearity: {self.linearity:.6f}")
        print(f"Planarity: {self.planarity:.6f}")
        print(f"Sphericity: {self.sphericity:.6f}")
        print(f"Omnivariance: {self.omnivariance:.6f}")
        print(f"Anisotropy: {self.anisotropy:.6f}")
        print(f"Eigenentropy: {self.eigenentropy:.6f}")
        print(f"Tortuosity: {self.tortuosity:.6f}")
        print(f"Curvature Change: {self.curvature_change:.6f}")

    @property
    def eigenvalues(self):
        """获取排序后的特征值"""
        return self._eigenvalues

    @property
    def eigenvectors(self):
        """获取排序后的特征向量"""
        return self._eigenvectors

    @property
    def all_measures(self):
        """获取所有特征值相关的计算结果"""
        return {
            "normalized_eigenvalues": {"e1": self.e1, "e2": self.e2, "e3": self.e3},
            "linearity": self.linearity,
            "planarity": self.planarity,
            "sphericity": self.sphericity,
            "omnivariance": self.omnivariance,
            "anisotropy": self.anisotropy,
            "eigenentropy": self.eigenentropy,
            "tortuosity": self.tortuosity,
            "curvature_change": self.curvature_change,
        }
