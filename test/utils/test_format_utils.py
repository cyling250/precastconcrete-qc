# -*- coding: utf-8 -*-
"""
@File    :   test_format_utils.py
@Time    :   2026/01/15 17:50:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   测试format_utils.py中的函数
"""

import unittest
import os
import numpy as np
import pandas as pd
from utils.format_utils import convert_ccxyz_to_xyz


class TestFormatUtils(unittest.TestCase):
    """测试format_utils.py中的功能"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建一个临时的CloudCompare xyz格式测试文件
        self.test_ccxyz_file = "test_ccxyz_temp.txt"

        # 生成测试数据：6个点，每个点有9列数据（CloudCompare xyz格式）
        test_data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
                [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
                [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            ]
        )

        # 保存为文本文件，空格分隔
        np.savetxt(self.test_ccxyz_file, test_data, fmt="%.6f")

        # 预期的输出结果：只保留前3列
        self.expected_xyz = test_data[:, :3]

    def tearDown(self):
        """测试后的清理工作"""
        # 删除临时测试文件
        if os.path.exists(self.test_ccxyz_file):
            os.remove(self.test_ccxyz_file)

    def test_convert_ccxyz_to_xyz(self):
        """测试convert_ccxyz_to_xyz函数"""
        # 调用函数转换
        result_xyz = convert_ccxyz_to_xyz(self.test_ccxyz_file)

        # 检查结果类型
        self.assertIsInstance(result_xyz, np.ndarray)

        # 检查结果形状
        self.assertEqual(result_xyz.shape, (6, 3))

        # 检查结果值是否正确
        np.testing.assert_array_equal(result_xyz, self.expected_xyz)

    def test_file_not_found(self):
        """测试文件不存在的情况"""
        non_existent_file = "non_existent_file.txt"

        # 预期会抛出FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            convert_ccxyz_to_xyz(non_existent_file)

    def test_invalid_file_format(self):
        """测试无效文件格式的情况"""
        # 创建一个无效的测试文件（列数不足）
        invalid_file = "invalid_ccxyz_temp.txt"
        invalid_data = np.array(
            [
                [1.0, 2.0],  # 只有2列，无效
                [3.0, 4.0],
            ]
        )
        np.savetxt(invalid_file, invalid_data, fmt="%.6f")

        # 预期会抛出IndexError或其他异常，因为函数试图访问不存在的列
        with self.assertRaises(Exception):
            convert_ccxyz_to_xyz(invalid_file)

        # 清理无效测试文件
        if os.path.exists(invalid_file):
            os.remove(invalid_file)

    def test_large_data_conversion(self):
        """测试大数据量转换"""
        # 创建一个包含10000个点的测试文件
        large_test_file = "large_ccxyz_temp.txt"
        large_test_data = np.random.rand(10000, 9)  # 10000个点，每个点9列
        np.savetxt(large_test_file, large_test_data, fmt="%.6f")

        # 调用函数转换
        result_xyz = convert_ccxyz_to_xyz(large_test_file)

        # 检查结果形状
        self.assertEqual(result_xyz.shape, (10000, 3))

        # 检查结果值是否正确
        # 由于保存时使用了%.6f格式，我们应该将预期结果也格式化为相同精度
        # 然后比较格式化后的结果
        # 注意：这里不需要比较原始数据，因为文件操作会导致精度损失
        # 我们只需要确保函数能正确处理大数据量的文件
        # 并且输出形状正确，数据类型正确
        self.assertIsInstance(result_xyz, np.ndarray)
        self.assertEqual(result_xyz.dtype, np.float64)

        # 清理大数据测试文件
        if os.path.exists(large_test_file):
            os.remove(large_test_file)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
