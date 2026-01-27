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
import open3d as o3d


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


def save_as_xyzc(pcd_data: list[o3d.geometry.PointCloud], file_path: str):
    """
    将XYZ坐标和点云类别保存为XYZC格式的文件。

    参数:
    pcd_data (list[o3d.geometry.PointCloud]): 包含XYZ坐标和类别信息的点云对象列表。
    file_path (str): 要保存的文件路径。
    """
    all_pcd_list = []
    for i in range(len(pcd_data)):
        inliers_pcd = o3d.geometry.PointCloud()
        inliers_pcd.points = o3d.utility.Vector3dVector(pcd_data[i])
        # 为每个点生成随机颜色
        random_color = np.random.rand(3)  # 生成随机RGB颜色
        inliers_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(random_color, (len(pcd_data[i]), 1))
        )
        all_pcd_list.append(inliers_pcd)
    # 初始化存储所有点和类别的列表
    all_points = []
    # 遍历每个类别的点云数据
    for class_id, pcd in enumerate(all_pcd_list):
        if pcd is None or len(pcd.points) == 0:
            continue
        # 获取点云坐标
        points = np.asarray(pcd.points)
        # 为当前类别的所有点分配类别ID
        classes = np.full((len(points), 1), class_id, dtype=int)
        # 合并坐标和类别
        xyzc = np.hstack((points, classes))
        all_points.append(xyzc)

    # 合并所有类别的数据
    if not all_points:
        print("警告：没有有效的点云数据可保存")
        return

    combined_data = np.vstack(all_points)

    # 保存为文本文件，格式为: x y z class
    np.savetxt(
        file_path,
        combined_data,
        fmt="%.6f %.6f %.6f %d",
    )
