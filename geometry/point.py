import numpy as np
from typing import Set


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
        # 1. 检查是否为None
        if other is None:
            return False
        # 2. 检查是否为同一类
        if self.__class__ != other.__class__:
            return False
        # 3. 两个点之间的距离是否小于1e-6
        return np.linalg.norm(self.location - other.location) < 1e-6


class CrossPoint(BasePoint):
    """交点模型"""

    def __init__(self, location: np.ndarray, *lines) -> None:
        """初始化交点模型

        Args:
            location (np.ndarray): 交点模型参数，形状为 (3,)，包含点的坐标 [x, y, z]
        """
        super().__init__(location)
        self.link_lines: Set = set(lines)  # 连接的交线
