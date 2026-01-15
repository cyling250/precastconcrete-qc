# -*- coding: utf-8 -*-
"""
@File    :   test_noise_utils.py
@Time    :   2026/01/15 17:55:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   测试noise_utils.py中的函数
"""

import unittest
import numpy as np
from utils.noise_utils import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_outlier_noise,
    add_surface_noise
)


class TestNoiseUtils(unittest.TestCase):
    """测试noise_utils.py中的功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建一个简单的点云测试数据
        self.point_cloud = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0]
        ])
        
        # 创建点云法线数据（用于表面噪声测试）
        self.normals = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ])
    
    def test_add_gaussian_noise(self):
        """测试添加高斯噪声"""
        # 复制原始点云，避免修改原始数据
        original_cloud = self.point_cloud.copy()
        
        # 添加高斯噪声
        noisy_cloud = add_gaussian_noise(original_cloud.copy(), mean=0, std=0.1)
        
        # 检查返回的点云形状是否与原始相同
        self.assertEqual(noisy_cloud.shape, original_cloud.shape)
        
        # 检查点云数据是否发生了变化
        self.assertFalse(np.array_equal(noisy_cloud, original_cloud))
        
        # 检查噪声是否在合理范围内
        max_diff = np.max(np.abs(noisy_cloud - original_cloud))
        self.assertLess(max_diff, 0.5)  # 99.7%的点应该在3倍标准差内（0.3），这里放宽到0.5
    
    def test_add_gaussian_noise_with_different_std(self):
        """测试不同标准差的高斯噪声"""
        original_cloud = self.point_cloud.copy()
        
        # 使用不同的标准差
        noisy_cloud_small_std = add_gaussian_noise(original_cloud.copy(), mean=0, std=0.01)
        noisy_cloud_large_std = add_gaussian_noise(original_cloud.copy(), mean=0, std=1.0)
        
        # 计算差异
        diff_small_std = np.mean(np.abs(noisy_cloud_small_std - original_cloud))
        diff_large_std = np.mean(np.abs(noisy_cloud_large_std - original_cloud))
        
        # 标准差越大，平均差异应该越大
        self.assertLess(diff_small_std, diff_large_std)
    
    def test_add_salt_pepper_noise(self):
        """测试添加椒盐噪声"""
        original_cloud = self.point_cloud.copy()
        
        # 添加椒盐噪声
        noisy_cloud = add_salt_pepper_noise(original_cloud.copy(), noise_ratio=0.2, salt_ratio=0.5)
        
        # 检查返回的点云形状是否与原始相同
        self.assertEqual(noisy_cloud.shape, original_cloud.shape)
        
        # 检查点云数据是否发生了变化
        self.assertFalse(np.array_equal(noisy_cloud, original_cloud))
        
        # 检查是否有足够的点被修改
        diff_points = np.sum(np.any(noisy_cloud != original_cloud, axis=1))
        # 噪声比例为0.2，应该有大约20%的点被修改
        self.assertGreaterEqual(diff_points, 1)  # 至少1个点被修改
    
    def test_add_salt_pepper_noise_ratio(self):
        """测试不同噪声比例的椒盐噪声"""
        original_cloud = self.point_cloud.copy()
        
        # 使用不同的噪声比例
        noisy_cloud_small_ratio = add_salt_pepper_noise(original_cloud.copy(), noise_ratio=0.1)
        noisy_cloud_large_ratio = add_salt_pepper_noise(original_cloud.copy(), noise_ratio=0.5)
        
        # 计算被修改的点数
        modified_points_small = np.sum(np.any(noisy_cloud_small_ratio != original_cloud, axis=1))
        modified_points_large = np.sum(np.any(noisy_cloud_large_ratio != original_cloud, axis=1))
        
        # 噪声比例越大，被修改的点应该越多
        self.assertLess(modified_points_small, modified_points_large)
    
    def test_add_outlier_noise(self):
        """测试添加离群点噪声"""
        original_cloud = self.point_cloud.copy()
        original_size = original_cloud.shape[0]
        
        # 添加离群点噪声
        noisy_cloud = add_outlier_noise(original_cloud.copy(), outlier_ratio=0.2, scale=1.0)
        
        # 检查返回的点云数量是否增加了
        self.assertGreater(noisy_cloud.shape[0], original_size)
        
        # 检查离群点数量是否符合预期
        expected_outliers = int(original_size * 0.2)
        actual_outliers = noisy_cloud.shape[0] - original_size
        self.assertEqual(actual_outliers, expected_outliers)
    
    def test_add_outlier_noise_ratio(self):
        """测试不同离群点比例"""
        original_cloud = self.point_cloud.copy()
        original_size = original_cloud.shape[0]
        
        # 使用不同的离群点比例
        noisy_cloud_small_ratio = add_outlier_noise(original_cloud.copy(), outlier_ratio=0.1)
        noisy_cloud_large_ratio = add_outlier_noise(original_cloud.copy(), outlier_ratio=0.5)
        
        # 计算离群点数量
        outliers_small = noisy_cloud_small_ratio.shape[0] - original_size
        outliers_large = noisy_cloud_large_ratio.shape[0] - original_size
        
        # 离群点比例越大，添加的离群点应该越多
        self.assertLess(outliers_small, outliers_large)
    
    def test_add_surface_noise(self):
        """测试添加表面噪声"""
        original_cloud = self.point_cloud.copy()
        
        # 添加表面噪声
        noisy_cloud = add_surface_noise(original_cloud.copy(), self.normals, std=0.1)
        
        # 检查返回的点云形状是否与原始相同
        self.assertEqual(noisy_cloud.shape, original_cloud.shape)
        
        # 检查点云数据是否发生了变化
        self.assertFalse(np.array_equal(noisy_cloud, original_cloud))
        
        # 检查噪声是否在合理范围内
        max_diff = np.max(np.abs(noisy_cloud - original_cloud))
        self.assertLess(max_diff, 0.5)  # 噪声应该在合理范围内
    
    def test_add_surface_noise_with_different_std(self):
        """测试不同标准差的表面噪声"""
        original_cloud = self.point_cloud.copy()
        
        # 使用不同的标准差
        noisy_cloud_small_std = add_surface_noise(original_cloud.copy(), self.normals, std=0.01)
        noisy_cloud_large_std = add_surface_noise(original_cloud.copy(), self.normals, std=1.0)
        
        # 计算差异
        diff_small_std = np.mean(np.abs(noisy_cloud_small_std - original_cloud))
        diff_large_std = np.mean(np.abs(noisy_cloud_large_std - original_cloud))
        
        # 标准差越大，平均差异应该越大
        self.assertLess(diff_small_std, diff_large_std)
    
    def test_normalized_normals(self):
        """测试法线是否被正确归一化"""
        # 创建非归一化法线
        non_normalized_normals = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0]
        ])
        
        # 复制原始点云的一部分
        cloud_part = self.point_cloud[:3].copy()
        
        # 添加表面噪声
        noisy_cloud = add_surface_noise(cloud_part, non_normalized_normals, std=0.1)
        
        # 检查是否能成功执行，没有异常
        self.assertIsInstance(noisy_cloud, np.ndarray)
    
    def test_empty_point_cloud(self):
        """测试空点云的情况"""
        empty_cloud = np.array([]).reshape(0, 3)
        empty_normals = np.array([]).reshape(0, 3)
        
        # 测试各种噪声函数是否能处理空点云
        # 高斯噪声
        result = add_gaussian_noise(empty_cloud.copy())
        self.assertEqual(result.shape, (0, 3))
        
        # 椒盐噪声
        result = add_salt_pepper_noise(empty_cloud.copy())
        self.assertEqual(result.shape, (0, 3))
        
        # 离群点噪声
        result = add_outlier_noise(empty_cloud.copy())
        # 空点云添加离群点后应该只包含离群点
        self.assertEqual(result.shape, (0, 3))  # 因为原始点云为空，outlier_ratio=0.1会生成0个离群点
        
        # 表面噪声
        result = add_surface_noise(empty_cloud.copy(), empty_normals)
        self.assertEqual(result.shape, (0, 3))


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
