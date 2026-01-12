# -*- coding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2025/12/24 14:42:58
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   测试脚本，用于测试点云数据处理工具
"""

import os
import sys

import open3d as o3d

file_path = "C:/Users/35160/Desktop/预制叠合梁300x600.ply"

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误：点云文件 '{file_path}' 不存在")
    sys.exit(1)

# 加载点云
print(f"正在加载点云文件：{file_path}...")
point_cloud = o3d.io.read_point_cloud(file_path)

# 检查点云是否成功加载
if point_cloud.is_empty():
    print("错误：点云文件为空或无法读取")
    sys.exit(1)

# 打印点云基本信息
point_count = len(point_cloud.points)
print("\n点云基本信息：")
print(f"- 点数量：{point_count}")

# 获取点云边界框
bounding_box = point_cloud.get_axis_aligned_bounding_box()
print("- 边界框：")
print(f"  最小坐标：{bounding_box.min_bound}")
print(f"  最大坐标：{bounding_box.max_bound}")

# 检查是否有颜色信息
if point_cloud.has_colors():
    print("- 包含颜色信息")
else:
    print("- 不包含颜色信息")

# 检查是否有法向量信息
if point_cloud.has_normals():
    print("- 包含法向量信息")
else:
    print("- 不包含法向量信息")

# 显示点云
print("\n显示点云...")
o3d.visualization.draw_geometries(
    [point_cloud], window_name="点云显示", width=800, height=600
)
