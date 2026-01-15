# -*- coding: utf-8 -*-
"""
@File    :   test_eigen_measures_utils.py
@Time    :   2026/01/15 17:45:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   测试eigen_measures_utils.py中的EigenMeasures类
"""

import unittest
import os
import numpy as np
import open3d as o3d
from utils.eigen_measures_utils import EigenMeasures


class TestEigenMeasures(unittest.TestCase):
    """测试EigenMeasures类的功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用用户提供的测试文件
        self.test_ply_file = "mock/res/point/预制叠合梁300x600.ply"
        
        # 确保测试文件存在
        self.assertTrue(os.path.exists(self.test_ply_file), f"测试文件不存在: {self.test_ply_file}")
        
        # 读取点云数据，用于测试不同输入类型
        self.pcd = o3d.io.read_point_cloud(self.test_ply_file)
        self.points = np.asarray(self.pcd.points)
    
    def test_init_with_file_path(self):
        """测试从文件路径初始化"""
        eigen_measures = EigenMeasures(self.test_ply_file)
        
        # 检查是否能正确计算特征值
        self.assertIsInstance(eigen_measures.eigenvalues, np.ndarray)
        self.assertEqual(eigen_measures.eigenvalues.shape, (3,))
        
        # 检查是否能正确访问各种特征指标
        self.assertIsInstance(eigen_measures.linearity, float)
        self.assertIsInstance(eigen_measures.planarity, float)
        self.assertIsInstance(eigen_measures.sphericity, float)
        self.assertIsInstance(eigen_measures.omnivariance, float)
        self.assertIsInstance(eigen_measures.anisotropy, float)
        self.assertIsInstance(eigen_measures.eigenentropy, float)
        self.assertIsInstance(eigen_measures.tortuosity, float)
        self.assertIsInstance(eigen_measures.curvature_change, float)
    
    def test_init_with_point_cloud_object(self):
        """测试从PointCloud对象初始化"""
        eigen_measures = EigenMeasures(self.pcd)
        
        # 检查是否能正确计算特征值
        self.assertIsInstance(eigen_measures.eigenvalues, np.ndarray)
        self.assertEqual(eigen_measures.eigenvalues.shape, (3,))
        
        # 检查特征指标是否为数值类型
        self.assertIsInstance(eigen_measures.linearity, float)
    
    def test_init_with_numpy_array(self):
        """测试从numpy数组初始化"""
        eigen_measures = EigenMeasures(self.points)
        
        # 检查是否能正确计算特征值
        self.assertIsInstance(eigen_measures.eigenvalues, np.ndarray)
        self.assertEqual(eigen_measures.eigenvalues.shape, (3,))
        
        # 检查特征指标是否为数值类型
        self.assertIsInstance(eigen_measures.planarity, float)
    
    def test_invalid_input_type(self):
        """测试无效输入类型"""
        with self.assertRaises(TypeError):
            EigenMeasures(123)  # 整数类型应该抛出异常
    
    def test_insufficient_points(self):
        """测试点云数据不足的情况"""
        # 创建只有2个点的点云
        few_points = np.array([[0, 0, 0], [1, 1, 1]])
        with self.assertRaises(ValueError):
            EigenMeasures(few_points)
    
    def test_all_measures_property(self):
        """测试all_measures属性"""
        eigen_measures = EigenMeasures(self.test_ply_file)
        all_measures = eigen_measures.all_measures
        
        # 检查返回值类型
        self.assertIsInstance(all_measures, dict)
        
        # 检查返回值包含所有预期的键
        expected_keys = [
            "normalized_eigenvalues",
            "linearity",
            "planarity",
            "sphericity",
            "omnivariance",
            "anisotropy",
            "eigenentropy",
            "tortuosity",
            "curvature_change"
        ]
        for key in expected_keys:
            self.assertIn(key, all_measures, f"all_measures中缺少键: {key}")
    
    def test_eigenvalues_property(self):
        """测试eigenvalues属性"""
        eigen_measures = EigenMeasures(self.test_ply_file)
        eigenvalues = eigen_measures.eigenvalues
        
        # 检查返回值类型和形状
        self.assertIsInstance(eigenvalues, np.ndarray)
        self.assertEqual(eigenvalues.shape, (3,))
        
        # 特征值应该是降序排列的
        self.assertGreaterEqual(eigenvalues[0], eigenvalues[1])
        self.assertGreaterEqual(eigenvalues[1], eigenvalues[2])
    
    def test_eigenvectors_property(self):
        """测试eigenvectors属性"""
        eigen_measures = EigenMeasures(self.test_ply_file)
        eigenvectors = eigen_measures.eigenvectors
        
        # 检查返回值类型和形状
        self.assertIsInstance(eigenvectors, np.ndarray)
        self.assertEqual(eigenvectors.shape, (3, 3))
    
    def test_normalized_eigenvalues(self):
        """测试归一化特征值"""
        eigen_measures = EigenMeasures(self.test_ply_file)
        
        # 检查归一化特征值之和接近1
        sum_eigenvalues = eigen_measures.e1 + eigen_measures.e2 + eigen_measures.e3
        self.assertAlmostEqual(sum_eigenvalues, 1.0, delta=1e-6)
    
    def test_feature_value_ranges(self):
        """测试特征值范围"""
        eigen_measures = EigenMeasures(self.test_ply_file)
        
        # 检查线性度范围 [0, 1]
        self.assertGreaterEqual(eigen_measures.linearity, 0.0)
        self.assertLessEqual(eigen_measures.linearity, 1.0)
        
        # 检查平面度范围 [0, 1]
        self.assertGreaterEqual(eigen_measures.planarity, 0.0)
        self.assertLessEqual(eigen_measures.planarity, 1.0)
        
        # 检查球形度范围 [0, 1]
        self.assertGreaterEqual(eigen_measures.sphericity, 0.0)
        self.assertLessEqual(eigen_measures.sphericity, 1.0)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
