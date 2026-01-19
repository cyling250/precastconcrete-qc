# -*- coding: utf-8 -*-
"""
@File    :   ransac_plane_fitting.py
@Time    :   2026/01/19 10:00:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   基于RANSAC算法的点云平面拟合
"""

import numpy as np
import open3d as o3d
from models.plane_fitting import RANSACPlaneFitting


def visualize_results(point_cloud_file, plane_fitter):
    """
    可视化拟合结果

    参数:
    point_cloud_file: str, 原始点云文件路径
    plane_fitter: RANSACPlaneFitting, 拟合后的平面拟合器
    """
    # 读取原始点云
    pcd = o3d.io.read_point_cloud(point_cloud_file)
    points = np.asarray(pcd.points)

    # 创建可视化对象
    vis_objects = []

    # 原始点云（灰色）
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = pcd.points
    pcd_original.colors = o3d.utility.Vector3dVector(
        np.zeros((points.shape[0], 3)) + 0.5
    )  # 灰色
    vis_objects.append(pcd_original)

    # 内点（绿色）
    if plane_fitter.best_inliers is not None:
        inlier_points = points[plane_fitter.best_inliers]
        pcd_inliers = o3d.geometry.PointCloud()
        pcd_inliers.points = o3d.utility.Vector3dVector(inlier_points)
        # 设置颜色
        inlier_colors = np.zeros((inlier_points.shape[0], 3))
        inlier_colors[:, 1] = 1.0  # 绿色
        pcd_inliers.colors = o3d.utility.Vector3dVector(inlier_colors)
        vis_objects.append(pcd_inliers)

    # 绘制平面
    if plane_fitter.best_plane is not None:
        # 计算平面的法向量和中心点
        normal = plane_fitter.get_normal_vector()
        center = (
            np.mean(points[plane_fitter.best_inliers], axis=0)
            if plane_fitter.best_inliers is not None
            else np.mean(points, axis=0)
        )

        # 计算平面的范围
        if plane_fitter.best_inliers is not None:
            inlier_points = points[plane_fitter.best_inliers]
            min_coords = np.min(inlier_points, axis=0)
            max_coords = np.max(inlier_points, axis=0)
            # 计算平面的尺寸
            width = max_coords[0] - min_coords[0]
            height = max_coords[1] - min_coords[1]
            # 如果宽度或高度为0，使用默认值
            if width < 1e-10 or height < 1e-10:
                width = 1.0
                height = 1.0
        else:
            # 使用默认尺寸
            width = 1.0
            height = 1.0

        # 创建平面（使用box创建）
        plane_mesh = o3d.geometry.TriangleMesh.create_box(
            width=width, height=height, depth=1e-10
        )

        # 移动平面到中心点
        plane_mesh.translate(np.array([width / 2, height / 2, 0]))

        # 计算旋转矩阵，使平面法向量与拟合平面法向量一致
        z_axis = np.array([0, 0, 1])

        # 如果法向量与z轴方向相反，调整法向量
        if np.dot(normal, z_axis) < 0:
            normal = -normal

        # 计算旋转轴和角度
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.arccos(
            np.dot(z_axis, normal) / (np.linalg.norm(z_axis) * np.linalg.norm(normal))
        )

        # 应用旋转
        if np.linalg.norm(rotation_axis) > 1e-10 and rotation_angle > 1e-10:
            plane_mesh.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rotation_axis * rotation_angle
                ),
                center=(0, 0, 0),
            )

        # 移动平面到点云的中心
        plane_mesh.translate(center - np.array([width / 2, height / 2, 0]))

        # 设置平面颜色（半透明蓝色）
        plane_mesh.paint_uniform_color([0, 0, 1])
        plane_mesh.compute_vertex_normals()
        vis_objects.append(plane_mesh)

    # 可视化
    print("正在可视化结果...")
    o3d.visualization.draw_geometries(
        vis_objects,
        window_name="RANSAC平面拟合结果",
        width=800,
        height=600,
        left=50,
        top=50,
    )


def main():
    """
    主函数，演示RANSAC平面拟合
    """
    import os

    # 点云文件路径
    # 如果没有提供点云文件，可以使用Open3D创建一个带有噪声的平面点云
    pcd_file = "mock/res/point/预制叠合梁300x600.ply"  # 替换为你的点云文件路径

    if os.path.exists(pcd_file):
        print(f"正在读取点云文件: {pcd_file}")
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)
    else:
        print(f"点云文件不存在，创建一个带有噪声的平面点云...")
        # 创建平面点云
        plane = o3d.geometry.Plane()
        pcd = plane.sample_points_uniformly(10000)
        points = np.asarray(pcd.points)

        # 添加噪声
        noise = np.random.normal(0, 0.05, points.shape)
        points += noise

    # 创建RANSAC平面拟合器
    plane_fitter = RANSACPlaneFitting(
        max_iterations=5000,
        distance_threshold=0.05,  # 根据点云数据调整
        confidence=0.99,
    )

    # 拟合平面
    plane, inliers = plane_fitter.fit(points)

    # 输出结果
    print(f"\n平面方程: {plane_fitter.get_plane_equation()}")
    print(f"法向量: {plane_fitter.get_normal_vector()}")
    print(f"内点数量: {len(inliers)}")
    print(f"外点数量: {points.shape[0] - len(inliers)}")
    print(f"内点比例: {len(inliers) / points.shape[0] * 100:.2f}%")

    # 可视化结果
    if os.path.exists(pcd_file):
        visualize_results(pcd_file, plane_fitter)
    else:
        # 可视化生成的点云
        vis_objects = []

        # 原始点云（灰色）
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(points)
        pcd_original.colors = o3d.utility.Vector3dVector(
            np.zeros((points.shape[0], 3)) + 0.5
        )  # 灰色
        vis_objects.append(pcd_original)

        # 内点（绿色）
        if inliers is not None:
            inlier_points = points[inliers]
            pcd_inliers = o3d.geometry.PointCloud()
            pcd_inliers.points = o3d.utility.Vector3dVector(inlier_points)
            # 设置颜色
            inlier_colors = np.zeros((inlier_points.shape[0], 3))
            inlier_colors[:, 1] = 1.0  # 绿色
            pcd_inliers.colors = o3d.utility.Vector3dVector(inlier_colors)
            vis_objects.append(pcd_inliers)

        # 可视化
        print("正在可视化结果...")
        o3d.visualization.draw_geometries(
            vis_objects,
            window_name="RANSAC平面拟合结果",
            width=800,
            height=600,
            left=50,
            top=50,
        )


if __name__ == "__main__":
    main()
