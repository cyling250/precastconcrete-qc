import os
import sys

# 添加项目根目录到Python的模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from models import PlaneCoordinateSystem


def create_single_plane_mesh(plane, pcd, color=[0.8, 0.2, 0.2]):
    """
    从平面方程+对应点云生成单个平面的初始面网格（点云约束自动边界）
    :param plane: 标准化后的平面方程(A,B,C,D)
    :param pcd: 该平面对应的Open3D点云对象（已预处理：去噪+下采样）
    :param color: 面网格颜色
    :return: 单个平面的面网格（TriangleMesh）、平面局部坐标系(u,v)、平面中心
    """
    A, B, C, D = plane
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)  # 平面中心（点云质心）
    n, u, v = PlaneCoordinateSystem.get_plane_local_coords(plane)  # 平面局部坐标系

    # 步骤1：点云投影到平面的二维局部坐标系
    points_centered = points - centroid
    u_vals = np.dot(points_centered, u)
    v_vals = np.dot(points_centered, v)
    points_2d = np.column_stack([u_vals, v_vals])

    # 步骤2：计算二维包围盒（确定平面有效边界，适配规则构件；凸包可选，适配异形）
    # x_min, y_min = np.min(points_2d, axis=0)
    # x_max, y_max = np.max(points_2d, axis=0)
    # 可选：用凸包替代包围盒（适配异形RC构件），注释掉包围盒，打开下面一行
    hull = ConvexHull(points_2d)
    hull_2d = points_2d[hull.vertices]
    x_min, y_min, x_max, y_max = (
        hull_2d[:, 0].min(),
        hull_2d[:, 1].min(),
        hull_2d[:, 0].max(),
        hull_2d[:, 1].max(),
    )

    # 边界向外扩展
    # 如果边界宽度大于1m，则向外扩展5cm，否则扩展0.1倍
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.05 if x_range > 1.0 else x_range * 0.1
    x_max += 0.05 if x_range > 1.0 else x_range * 0.1
    y_min -= 0.05 if y_range > 1.0 else y_range * 0.1
    y_max += 0.05 if y_range > 1.0 else y_range * 0.1

    # 步骤3：生成二维矩形轮廓（有效边界）
    rect_2d = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    # 步骤4：二维轮廓还原为三维坐标
    rect_3d = centroid + rect_2d[:, 0:1] * u + rect_2d[:, 1:2] * v

    # 步骤5：三角化生成面网格（矩形→2个三角形）
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(rect_3d)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh, (u, v), centroid
