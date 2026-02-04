import numpy as np
from .plane import BasePlane
from .line import BaseLine
from .base_utils import SVD


def calc_inliers_plane(
    plane: BasePlane, points: np.ndarray, dst_threshold: float
) -> np.ndarray:
    """根据点到平面的距离计算内点和外点

    Args:
        points (np.ndarray): 输入点云，形状为 (N, 3)
        dst_threshold (float): 内点距离阈值

    Returns:
        np.ndarray: 内点索引，形状为 (M,)
    """
    if plane.parameters is None:
        raise ValueError("平面模型参数未初始化")
    c = plane.parameters[0:3]
    n = plane.parameters[3:6]
    dst = abs(np.dot(points - c, n))
    ind = dst < dst_threshold
    return ind


def calc_inliers_line(
    line: BaseLine, points: np.ndarray, dst_threshold: float
) -> np.ndarray:
    """
    根据点到直线的距离计算内点和外点

    参数:
        line: BaseLine, 直线模型
        points: numpy.ndarray, 形状为 (n_points, 3)，三维点云坐标
        dst_threshold: float, 距离阈值，小于该阈值的点被认为是内点

    返回:
        numpy.ndarray, 形状为 (n_points,)的布尔数组，True表示内点，False表示外点
    """

    c = line.parameters[0:3]  # 直线上的一点
    n = line.parameters[3:6]  # 直线的方向向量

    # 计算点到直线的距离
    # 距离公式: |(p - c) × n| / |n| 由于n已经归一化，所以分母为1
    dst = np.linalg.norm(np.cross(points - c, n), axis=1)
    ind = dst < dst_threshold
    return ind


def estimate_plane(pts: np.ndarray) -> BasePlane:
    """根据输入点云估计平面模型

    Args:
        pts (np.ndarray): 输入点云，形状为 (N, 3)

    Returns:
        BasePlane: 平面模型参数，形状为 (6,)
    """
    num = pts.shape[0]
    if num == 3:
        c = np.mean(pts, axis=0)
        l1 = pts[1] - pts[0]
        l2 = pts[2] - pts[0]
        n = np.cross(l1, l2)
        scale = [n[i] ** 2 for i in range(n.shape[0])]
        n = n / np.sqrt(np.sum(scale))
    else:
        _, _, c, n = SVD(pts)

    params = np.hstack((c.reshape(1, -1), n.reshape(1, -1)))[0, :]
    return BasePlane(params)


def estimate_line(pts: np.ndarray, control_line: np.ndarray = None) -> BaseLine:
    """
    估算直线模型参数，使用最小二乘法拟合点集

    参数:
        pts: numpy.ndarray, 形状为 (n_points, 3)，用于拟合直线的点集
        control_line: 可选参数，当提供控制直线时，所得到的拟合直线一定与该直线相交形状为 (6,)的数组，包含控制直线参数 [cx, cy, cz, nx, ny, nz]
        - 前3个参数: 控制直线上的一点坐标
        - 后3个参数: 控制直线的方向向量（已归一化）

    返回:
        numpy.ndarray, 形状为 (6,)的数组，包含直线参数 [cx, cy, cz, nx, ny, nz]
        - 前3个参数: 直线上的一点坐标
        - 后3个参数: 直线的方向向量（已归一化）
    """
    # 最小二乘法估算直线模型
    # 只有两个点时，可以直接计算
    num = pts.shape[0]
    if num == 2:
        # 两点确定一条直线
        c = np.mean(pts, axis=0)  # 直线的中点
        n = pts[1] - pts[0]  # 直线的方向向量
        # 归一化方向向量
        n = n / np.linalg.norm(n)
    else:
        # 使用SVD方法拟合直线
        # 对于直线拟合，我们需要找到点云的主方向
        u, _, c, _ = SVD(pts)
        # 主方向是最大特征值对应的特征向量
        n = u[:, 0]
        # 确保方向向量归一化
        n = n / np.linalg.norm(n)

    # 如果提供了控制直线，确保拟合直线与控制直线共面
    if control_line is not None:
        # 获取控制直线的参数
        c_control = control_line[0:3]  # 控制直线上的一点
        n_control = control_line[3:6]  # 控制直线的方向向量（已归一化）

        # 计算共面平面的法向量：两个方向向量的叉积
        plane_normal = np.cross(n, n_control)

        # 如果两个向量平行（叉积接近零），则直接返回
        if np.linalg.norm(plane_normal) < 1e-10:
            params = np.hstack((c.reshape(1, -1), n.reshape(1, -1)))[0, :]
            return BaseLine(params)
        else:
            # 归一化共面平面法向量
            plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # 将点c向共面平面投影
        # 平面方程: (x - c_control) · plane_normal = 0
        # 投影公式: c_projected = c - [(c - c_control) · plane_normal] * plane_normal
        projection_vector = np.dot(c - c_control, plane_normal) * plane_normal
        c_projected = c - projection_vector

        # 使用投影后的点作为新的直线点
        c = c_projected

    params = np.hstack((c.reshape(1, -1), n.reshape(1, -1)))[0, :]
    return BaseLine(params)
