# -*- coding: utf-8 -*-
"""
@File    :   format_utils.py
@Time    :   2026/01/13 15:18:27
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   None
"""

import numpy as np
import pandas as pd


def convert_ccxyz_to_xyz(cc_file: str) -> np.ndarray:
    """
    将CloudCompare导出的xyz格式文件转换为纯XYZ格式的点云数组。

    参数:
    cc_file (str): CloudCompare导出的xyz文件的路径。

    返回:
    np.ndarray: 包含XYZ坐标的点云数组，形状为 (N, 3)，其中N是点的数量。
    """
    # 读取CloudCompare导出的xyz文件 空格分割的文本文件
    df = pd.read_csv(
        cc_file,
        sep=" ",  # 使用任意数量的空白字符作为分隔符
        header=None,  # 没有表头
        dtype=np.float64,  # 数据类型为浮点数
    )
    # 删除3，4，5列
    df = df.drop(columns=[3, 4, 5, 6, 7, 8])
    # 将123列转换为Numpy数组
    xyz = df[[0, 1, 2]].to_numpy()
    return xyz
