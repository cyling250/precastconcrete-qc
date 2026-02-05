import numpy as np
from .line import BaseLine, CrossLine
from .plane import BasePlane
from .point import BasePoint
from .base_utils import project_points_to_line


def intersect_plane(
    plane1: BasePlane,
    plane2: BasePlane,
) -> CrossLine | None:
    """计算当前平面与另一个平面的交线

    Args:
        plane1 (BasePlane): 第一个平面模型
        plane2 (BasePlane): 第二个平面模型
    Returns:
        CrossLine | None: 两个平面的交线模型
    """
    # 获取两个平面的参数
    c1 = plane1.parameters[0:3]  # 平面1上的点
    n1 = plane1.parameters[3:6]  # 平面1的法向量
    c2 = plane2.parameters[0:3]  # 平面2上的点
    n2 = plane2.parameters[3:6]  # 平面2的法向量

    # 计算交线的方向向量（两个法向量的叉积）
    direction = np.cross(n1, n2)

    # 检查平面是否平行
    if np.linalg.norm(direction) < 1e-6:
        return None

    # 归一化方向向量
    direction = direction / np.linalg.norm(direction)

    # 计算交线上的一个点
    # 使用线性代数方法求解交线上的点
    # 我们需要解方程组：
    # n1 · (p - c1) = 0
    # n2 · (p - c2) = 0
    # 构建系数矩阵
    A = np.vstack([n1, n2])
    b = np.array([np.dot(n1, c1), np.dot(n2, c2)])

    # 使用最小二乘法求解
    point_on_line, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # 创建LineModel参数：点 + 方向向量
    line_params = np.hstack([point_on_line, direction])
    cross_line = CrossLine(line_params, plane1, plane2)

    return cross_line


def intersect_plane_with_pcd(
    plane1: BasePlane,
    plane2: BasePlane,
    pcd1: np.ndarray,
    pcd2: np.ndarray,
) -> CrossLine | None:
    """计算当前平面与点云的交线,并根据点云参数设置交线的起终点

    Args:
        plane1 (BasePlane): 第一个平面模型
        plane2 (BasePlane): 第二个平面模型
        pcd1 (np.ndarray): 第一个平面上的点云，形状为 (N, 3)
        pcd2 (np.ndarray): 第二个平面上的点云，形状为 (N, 3)

    Returns:
        CrossLine | None: 平面与点云的交线模型
    """
    cross_line = intersect_plane(plane1, plane2)
    if cross_line is None:
        return None
    # 判断点云是否相交 使用AABB包围盒区间重叠法
    max_pcd1 = np.max(pcd1, axis=0)
    min_pcd1 = np.min(pcd1, axis=0)
    max_pcd2 = np.max(pcd2, axis=0)
    min_pcd2 = np.min(pcd2, axis=0)
    # 包围盒向外扩展1cm 只扩展其中一个
    max_pcd1 += 0.01
    min_pcd1 -= 0.01
    max_pcd2 += 0.01
    min_pcd2 -= 0.01
    for i in range(3):
        if max_pcd1[i] < min_pcd2[i] or max_pcd2[i] < min_pcd1[i]:
            return None

    P0 = cross_line.parameters[0:3]  # 交线上的点
    V = cross_line.parameters[3:6]  # 交线方向向量
    proj_pcd1 = project_points_to_line(pcd1, P0, V)
    proj_pcd2 = project_points_to_line(pcd2, P0, V)

    t_pcd1 = cross_line.get_point_lambda(proj_pcd1)
    t_pcd2 = cross_line.get_point_lambda(proj_pcd2)

    # 计算两个投影点云在交线上的参数范围
    t_range1_min = np.min(t_pcd1)
    t_range1_max = np.max(t_pcd1)
    t_range2_min = np.min(t_pcd2)
    t_range2_max = np.max(t_pcd2)

    # 计算两个参数范围的交集
    t_intersect_min = np.max([t_range1_min, t_range2_min])
    t_intersect_max = np.min([t_range1_max, t_range2_max])

    # 如果有交集，设置交线的起终点
    if t_intersect_min <= t_intersect_max:
        point_start = P0 + t_intersect_min * V
        point_end = P0 + t_intersect_max * V
        cross_line.point_start = point_start
        cross_line.point_end = point_end
    else:
        # 无交集，返回None
        return None
    return cross_line


def intersect_line(
    line1: BaseLine | CrossLine,
    line2: BaseLine | CrossLine,
    eps: float = 1e-6,
    expand: float = 0.01,
) -> BasePoint | None:
    """
    计算与另一条直线的交点（基于知乎几何算法）

    参数:
        line1: CrossLine | BaseLine, 直线1模型
        line2: CrossLine | BaseLine, 直线2模型
        eps: float = 1e-6, 精度参数，默认值为 1e-6
        expand: float = 0.01, 扩展参数，默认值为 0.01

    返回:
        BasePoint | None, 交点对象，若直线平行或不相交则返回 None
    """
    # 解包直线参数
    x1, y1, z1, a1, b1, c1 = line1.parameters
    x2, y2, z2, a2, b2, c2 = line2.parameters

    # 向量P1->P2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # 步骤1：计算方向向量的叉乘 v1×v2
    v1_cross_v2 = np.cross([a1, b1, c1], [a2, b2, c2])
    if np.linalg.norm(v1_cross_v2) < eps:
        # 平行时，判断P2是否在L1上（即向量P1P2与v1平行）
        p1p2_cross_v1 = np.cross([dx, dy, dz], [a1, b1, c1])
        if np.linalg.norm(p1p2_cross_v1) < eps:
            # 平行且重合，有无穷多交点
            return None
        else:
            # 平行但不重合，无交点
            return None

    # 步骤2：判断是否共面：混合积(P1P2 · (v1×v2))的绝对值小于eps则共面
    mix_product = np.dot([dx, dy, dz], v1_cross_v2)
    if abs(mix_product) > 1e-6:
        # 异面直线，无交点
        return None

    # 步骤3：基于MATLAB几何算法求解交点
    # 直线1方向向量（已归一化）
    v1 = np.array([a1, b1, c1])
    v1_norm = np.linalg.norm(v1)
    v1_dir = v1 / v1_norm

    # 直线2方向向量（已归一化）
    v2 = np.array([a2, b2, c2])
    v2_norm = np.linalg.norm(v2)
    v2_dir = v2 / v2_norm

    # P1P2向量
    p1p2_vec = np.array([dx, dy, dz])

    # 计算点P2到直线L1的距离
    p2p3_norm = np.linalg.norm(np.cross(p1p2_vec, v1_dir))

    # 投影点P3坐标（P2在L1上的投影）
    p3_pnt = np.array([x1, y1, z1]) + np.dot(p1p2_vec, v1_dir) * v1_dir

    # 计算两直线方向向量的夹角余弦值
    cos_theta = abs(np.dot(v1_dir, v2_dir))

    intersection_point = None

    if cos_theta < 1e-7:
        # 两直线垂直，交点就是投影点P3
        intersection_point = p3_pnt
    else:
        # 计算夹角正切值
        tan_theta = np.sqrt(1 - cos_theta**2) / cos_theta

        # 计算KP3的距离
        kp3_norm = p2p3_norm / tan_theta

        # 计算两个可能的交点
        k1_pnt = p3_pnt + kp3_norm * v1_dir
        k2_pnt = p3_pnt - kp3_norm * v1_dir

        # 计算两个交点到直线L2的距离
        p2k1_vec = k1_pnt - np.array([x2, y2, z2])
        p2k2_vec = k2_pnt - np.array([x2, y2, z2])

        d1 = np.linalg.norm(np.cross(p2k1_vec, v2_dir))  # 点K1到直线L2的距离
        d2 = np.linalg.norm(np.cross(p2k2_vec, v2_dir))  # 点K2到直线L2的距离

        # 选择距离小的点作为交点
        if d1 < d2:
            intersection_point = k1_pnt
        else:
            intersection_point = k2_pnt

    # 交线的交点需要在交线的范围内
    if type(line1) is CrossLine:
        t1 = line1.get_point_lambda(intersection_point)
        range_line1 = [
            line1.get_point_lambda(line1.point_start),
            line1.get_point_lambda(line1.point_end),
        ]
        if min(range_line1) - expand > t1 or t1 > max(range_line1) + expand:
            return None

    if type(line2) is CrossLine:
        t2 = line2.get_point_lambda(intersection_point)
        range_line2 = [
            line2.get_point_lambda(line2.point_start),
            line2.get_point_lambda(line2.point_end),
        ]
        if min(range_line2) - expand > t2 or t2 > max(range_line2) + expand:
            return None

    return BasePoint(intersection_point)
