from typing import Set
from typing import List
import numpy as np


class BaseLine(object):
    def __init__(self, params: np.ndarray):
        # 线段参数（线段上任意一点 cx, cy, cx） + （线段方向向量 nx, ny, nz）
        if params is None or type(params) is not np.ndarray or params.shape[0] != 6:
            raise AttributeError("直线模型参数必须是六维ndarray数组")
        else:
            self._parameters = params
            # 归一化方向向量
            self._parameters[3:6] = self._parameters[3:6] / np.linalg.norm(
                self._parameters[3:6]
            )

    def __str__(self) -> str:
        """返回直线模型的字符串表示"""
        return f"BaseLine({self._parameters})"

    def get_parameters(self) -> np.ndarray:
        """返回直线模型的参数"""
        return self._parameters


class CrossLine(BaseLine):
    """由两个平面相交所得到的直线模型"""

    from .plane_model import BasePlane

    def __init__(self, params: np.ndarray, plane1: BasePlane, plane2: BasePlane):
        super().__init__(params)
        from .plane_model import BasePlane

        if plane1 is None or plane2 is None:
            raise AttributeError("交线模型必须包含两个平面模型")
        self.plane1: BasePlane = plane1
        self.plane2: BasePlane = plane2

    def in_cloud(self, points: np.ndarray, dst_threshold: float = 0.05) -> bool:
        """判断交线是否在点云中

        Args:
            points (np.ndarray): 输入点云，形状为 (N, 3)
            dst_threshold (float, optional): 距离阈值，默认值为 0.01

        Returns:
            bool: True表示交线在点云中，False表示交线不在点云中
        """
        # 1. 将点云向self._plane1投影
        from .plane_model import project2plane

        projected_points = project2plane(self.plane1, points)

        # 获取交线参数
        P0 = self._parameters[0:3]  # 交线起点
        V = self._parameters[3:6]  # 交线方向向量

        # 2. 判断点云在直线的哪一边
        # 生成点云中的点到直线上该点的向量
        vectors_to_line = projected_points - P0

        # 用该向量与直线的方向向量做叉乘
        cross_products = np.cross(vectors_to_line, V)

        # 核对叉乘结果是否在同一边
        # 叉乘结果的符号表示点在直线的哪一侧
        # 如果所有叉乘结果的某个分量（如z分量）符号相同，则点在同一边
        cross_z_signs = np.sign(cross_products[:, 2])  # 取z分量符号
        unique_signs = np.unique(cross_z_signs)

        if len(unique_signs) > 1:
            # 点云分布在直线两侧，返回True
            return True
        else:
            # 点云都在直线同一侧，判断距离
            # 计算每个点到交线的最短距离
            distances = np.linalg.norm(cross_products, axis=1) / np.linalg.norm(V)
            # 判断最近的点是否小于阈值
            min_distance = np.min(distances)
            return min_distance <= dst_threshold

    def __eq__(self, other: "CrossLine") -> bool:
        """判断两个交线是否相等"""
        # 两个交线相等，它们的plane1和plane2必须相等或者它们的plane1和plane2交换顺序后相等
        if self.plane1 == other.plane1 and self.plane2 == other.plane2:
            return True
        if self.plane1 == other.plane2 and self.plane2 == other.plane1:
            return True
        return False

    def __hash__(self):
        plane_set = (self.plane1.__hash__(), self.plane2.__hash__())
        return hash(plane_set)


class CrossPoint:
    """交点模型"""

    def __init__(
        self, location: np.ndarray, line1: CrossLine, line2: CrossLine
    ) -> None:
        """初始化交点模型

        Args:
            location (np.ndarray): 交点模型参数，形状为 (3,)，包含点的坐标 [x, y, z]
        """
        self.location = location
        self.link_lines: Set[CrossLine] = set([line1, line2])  # 连接的交线

    def equal(self, other: "CrossPoint") -> bool:
        """判断两个交点是否相等"""
        # 1. 检查是否为None
        if other is None:
            return False
        # 2. 检查是否为同一类
        if self.__class__ != other.__class__:
            return False
        # 3. 两个点之间的距离是否小于1e-6
        return np.linalg.norm(self.location - other.location) < 1e-6

    def get_location(self) -> np.ndarray:
        """返回交点模型的参数"""
        return self.location


def SVD(points):
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c  # shift the points
    A = A.T  # 3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)  # A=u*s*vh
    normal = u[:, -1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal, normal))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u, s, c, normal


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

    c = line.get_parameters()[0:3]  # 直线上的一点
    n = line.get_parameters()[3:6]  # 直线的方向向量

    # 计算点到直线的距离
    # 距离公式: |(p - c) × n| / |n| 由于n已经归一化，所以分母为1
    dst = np.linalg.norm(np.cross(points - c, n), axis=1)
    ind = dst < dst_threshold
    return ind


def estimate_line(pts: np.ndarray, control_line: np.ndarray = None) -> BaseLine:
    """
    估算直线模型

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


def intersect_line(
    line1: BaseLine, line2: BaseLine, eps: float = 1e-6
) -> CrossPoint | None:
    """
    计算与另一条直线的交点（基于知乎几何算法）

    参数:
        line1: CrossLine | BaseLine, 直线1模型
        line2: CrossLine | BaseLine, 直线2模型

    返回:
        CrossPoint | None, 交点对象，若直线平行或不相交则返回 None
    """
    # 核心修改：通过get_parameters()解包直线参数
    x1, y1, z1, a1, b1, c1 = line1.get_parameters()
    x2, y2, z2, a2, b2, c2 = line2.get_parameters()

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

    return CrossPoint(intersection_point, line1, line2)
