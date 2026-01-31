import os
import sys

# 添加项目根目录到Python的模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from models import LineModel, PlaneModel


def line_in_pcd(
    line: LineModel,
    plane: PlaneModel,
    pcd,
    threshold: float = 0.05,
):
    """
    判断交线是否在点云内

    Args:
        line (LineModel): 交线模型
        plane (PlaneModel): 平面模型
        pcd: 点云模型
        threshold (float): 距离阈值，默认0.01

    Returns:
        bool: 如果点云全部在交线的同一边且距离大于阈值返回False，否则返回True
    """
    assert line.parameters is not None, "交线参数不能为空"
    assert plane.parameters is not None, "平面参数不能为空"

    # 1. 确保交线在平面上
    # 获取交线上的一个点和方向向量
    line_point = line.parameters[0:3]
    line_direction = line.parameters[3:6]

    # 获取平面参数
    plane_point = plane.parameters[0:3]
    plane_normal = plane.parameters[3:6]

    # 检查交线是否在平面上
    # 交线上的点应该在平面上
    distance_to_plane = abs(np.dot(line_point - plane_point, plane_normal))
    if distance_to_plane > 1e-6:
        logging.error("交线不在平面上")
        return False

    # 交线方向应该与平面法向量垂直
    dot_product = np.dot(line_direction, plane_normal)
    if abs(dot_product) > 1e-6:
        logging.error("交线方向不与平面垂直")
        return False

    # 2. 将点云投影到平面上
    points = np.asarray(pcd.points)
    projected_points = plane.project_points(points)

    # 3. 判断点云是否全部在交线的同一边且距离大于阈值
    # 计算每个点到交线的距离
    distances = []
    sides = []

    for point in projected_points:
        # 计算点到交线的向量
        vector_to_point = point - line_point

        # 计算点到交线的垂直距离
        # 距离 = |(point - line_point) × line_direction| / |line_direction|
        cross_product = np.cross(vector_to_point, line_direction)
        distance = np.linalg.norm(cross_product)
        distances.append(distance)

        # 判断点在交线的哪一边
        # 使用平面法向量和交线方向向量的叉积作为参考方向
        reference_vector = np.cross(plane_normal, line_direction)
        side = np.sign(np.dot(vector_to_point, reference_vector))
        sides.append(side)

    distances = np.array(distances)
    sides = np.array(sides)

    # 检查是否所有点都在同一边
    unique_sides = np.unique(sides)
    if len(unique_sides) == 1:
        # 所有点都在同一边，检查距离是否都大于阈值
        if np.all(distances > threshold):
            return False  # 所有点都在同一边且距离大于阈值
        else:
            return True  # 所有点都在同一边但有些点距离小于等于阈值
    else:
        return True  # 点分布在交线两侧
