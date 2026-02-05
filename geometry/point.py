import numpy as np


class BasePoint:
    """基础点模型"""

    def __init__(self, location: np.ndarray) -> None:
        """初始化基础点模型

        Args:
            location (np.ndarray): 基础点模型参数，形状为 (3,)，包含点的坐标 [x, y, z]
        """
        self.location = location

    def __eq__(self, other: "BasePoint") -> bool:
        """判断两个点是否相等"""
        # 检查是否为同一类
        if self.__class__ != other.__class__:
            return False
        # 两个点之间的距离是否小于1e-6
        return np.linalg.norm(self.location - other.location) < 1e-6

    def __sub__(self, other) -> np.ndarray:
        """计算两个点之间的向量"""
        if isinstance(other, BasePoint):
            return self.location - other.location
        elif isinstance(other, np.ndarray):
            return self.location - other
        else:
            raise TypeError(f"不支持的操作数类型{type(other)}")

    def __str__(self) -> str:
        """返回点的字符串表示"""
        return f"BasePoint({self.location})"
