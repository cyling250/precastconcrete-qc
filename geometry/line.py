from deprecated import deprecated
import numpy as np


class BaseLine(object):
    def __init__(self, params: np.ndarray):
        # 线段参数（线段上任意一点 cx, cy, cx） + （线段方向向量 nx, ny, nz）
        if params is None or type(params) is not np.ndarray or params.shape[0] != 6:
            raise AttributeError("直线模型参数必须是六维ndarray数组")
        else:
            self.parameters = params
            # 归一化方向向量
            self.parameters[3:6] = self.parameters[3:6] / np.linalg.norm(
                self.parameters[3:6]
            )

    def __str__(self) -> str:
        """返回直线模型的字符串表示"""
        return f"BaseLine({self.parameters})"


class CrossLine(BaseLine):
    """由两个平面相交所得到的线段模型"""

    def __init__(
        self,
        params: np.ndarray,
        plane1,
        plane2,
        point_start: np.ndarray = None,
        point_end: np.ndarray = None,
    ):
        super().__init__(params)

        self.plane1 = plane1  # 交线所在的第一个平面
        self.plane2 = plane2  # 交线所在的第二个平面
        self.point_start: np.ndarray = None  # 交线起点 由点云投影得到大致位置
        self.point_end: np.ndarray = None  # 交线终点

    def __eq__(self, other: "CrossLine") -> bool:
        """判断两个交线是否相等"""
        # 两个交线相等，它们的plane1和plane2必须相等或者它们的plane1和plane2交换顺序后相等
        if self.plane1 == other.plane1 and self.plane2 == other.plane2:
            return True
        if self.plane1 == other.plane2 and self.plane2 == other.plane1:
            return True
        return False

    def __hash__(self):
        """返回交线的哈希值"""
        plane_set = (self.plane1.__hash__(), self.plane2.__hash__())
        return hash(plane_set)

    @deprecated
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
        P0 = self.parameters[0:3]  # 交线起点
        V = self.parameters[3:6]  # 交线方向向量

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

    def get_line_parameters(self, points: np.ndarray):
        """计算每个点在交线上的参数t"""
        # 校验点在线上
        if self.parameters is None:
            raise ValueError("交线模型参数未初始化")
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        # 计算点到直线的距离
        distances = np.linalg.norm(np.cross(points - c, n), axis=1) / np.linalg.norm(n)
        # 检查所有点是否都在线上
        if np.any(distances > 1e-6):
            raise ValueError("点不在交线上")
        # 计算t值
        vectors = points - self.parameters[0:3]
        t = np.dot(vectors, self.parameters[3:6])
        return t
