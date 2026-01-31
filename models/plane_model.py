from utils.math_utils import SVD
import numpy as np
from .line_model import LineModel


class PlaneModel(object):
    def __init__(self, params: np.ndarray | None = None):
        # 平面模型参数（平面上任意一点 cx, cy, cx） + （平面法向量 nx, ny, nz）
        self.parameters: np.ndarray | None = params

    def calc_inliers(self, points, dst_threshold):
        """根据点到平面的距离计算内点和外点

        Args:
            points (np.ndarray): 输入点云，形状为 (N, 3)
            dst_threshold (float): 内点距离阈值

        Returns:
            np.ndarray: 内点索引，形状为 (M,)
        """
        if self.parameters is None:
            raise ValueError("平面模型参数未初始化")
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        dst = abs(np.dot(points - c, n))
        ind = dst < dst_threshold
        return ind

    def estimate_parameters(self, pts):
        """根据输入点云估计平面模型参数

        Args:
            pts (np.ndarray): 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 平面模型参数，形状为 (6,)
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
        self.parameters = params
        return params

    def project_points(self, points):
        """将所有点向平面投影

        Args:
            points (np.ndarray): 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 投影点云，形状为 (N, 3)
        """
        if self.parameters is None:
            raise ValueError("平面模型参数未初始化")
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        proj_points = points - np.dot(points - c, n.reshape(-1, 1)) * n.reshape(1, -1)
        return proj_points

    def intersect(self, other) -> LineModel:
        """计算当前平面与另一个平面的交线

        Args:
            other (PlaneModel): 另一个平面模型

        Returns:
            LineModel: 两个平面的交线模型

        Raises:
            ValueError: 当平面平行时，抛出异常
        """
        assert self.parameters is not None and other.parameters is not None, (
            "平面模型参数未初始化"
        )

        # 获取两个平面的参数
        c1 = self.parameters[0:3]  # 平面1上的点
        n1 = self.parameters[3:6]  # 平面1的法向量
        c2 = other.parameters[0:3]  # 平面2上的点
        n2 = other.parameters[3:6]  # 平面2的法向量

        # 计算交线的方向向量（两个法向量的叉积）
        direction = np.cross(n1, n2)

        # 检查平面是否平行
        if np.linalg.norm(direction) < 1e-6:
            raise ValueError("平面平行，无交线")

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

        return LineModel(line_params)

    def get_standard(self):
        """获取平面的标准形式参数Ax+By+Cz+D=0

        Returns:
            np.ndarray: 标准形式参数，形状为 (4,)
        """
        if self.parameters is None:
            raise ValueError("平面模型参数未初始化")
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        d = -np.dot(c, n)
        return np.hstack((n, d))
