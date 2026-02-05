from deprecated import deprecated
import numpy as np


class BaseLine(object):
    def __init__(self, params: np.ndarray, **kwargs):
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

    def get_point_lambda(self, points):
        """计算每个点在交线上的参数t"""
        # 校验点在线上
        if self.parameters is None:
            raise ValueError("交线模型参数未初始化")
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        # 计算点到直线的距离
        distances = np.linalg.norm(np.cross(points - c, n)) / np.linalg.norm(n)
        # 检查所有点是否都在线上
        if np.any(distances > 1e-6):
            raise ValueError("点不在交线上")
        # 计算t值
        vectors = points - self.parameters[0:3]
        t = np.dot(vectors, self.parameters[3:6])
        return t


class SegmentLine(BaseLine):
    """由线段上的两个点所得到的线段模型"""

    def __init__(
        self,
        params: np.ndarray,
        link_points: list["BasePoint"] = None,
        **kwargs,
    ):
        super().__init__(params)
        if link_points is None:
            self.link_points = [None, None]
        else:
            self.link_points = link_points

    def __eq__(self, other: "SegmentLine") -> bool:
        """判断两个线段是否相等"""
        return (
            self.link_points[0] == other.link_points[0]
            and self.link_points[1] == other.link_points[1]
        ) or (
            self.link_points[0] == other.link_points[1]
            and self.link_points[1] == other.link_points[0]
        )

    def add_link_point(self, point: "BasePoint"):
        """添加线段的一个端点"""
        # 校验点在直线上
        self.get_point_lambda(point.location)
        # 校验点是否已经添加
        if point in self.link_points:
            return
        if self.link_points[0] is None:
            self.link_points[0] = point
        elif self.link_points[1] is None:
            self.link_points[1] = point
        else:
            raise ValueError("线段已经有两个端点")

    def __str__(self) -> str:
        """返回线段模型的字符串表示"""
        return f"SegmentLine({self.parameters}, {self.link_points})"


class CrossLine(BaseLine):
    """由两个平面相交所得到的线段模型"""

    def __init__(
        self,
        params: np.ndarray,
        plane1: "BasePlane",
        plane2: "BasePlane",
        point_start: np.ndarray = None,
        point_end: np.ndarray = None,
        **kwargs,
    ):
        super().__init__(params)

        self.plane1 = plane1  # 交线所在的第一个平面
        self.plane2 = plane2  # 交线所在的第二个平面
        self.point_start: np.ndarray = point_start  # 交线起点 由点云投影得到大致位置
        self.point_end: np.ndarray = point_end  # 交线终点 由点云投影得到大致位置

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


if __name__ == "__main__":
    from .point import BasePoint
    from .plane import BasePlane
