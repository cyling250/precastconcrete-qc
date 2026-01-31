import numpy as np


class PlaneCoordinateSystem:
    """
    平面坐标系类，用于生成和管理平面局部二维坐标系
    """
    
    @staticmethod
    def get_plane_local_coords(plane):
        """
        从平面方程生成局部二维坐标系（u,v轴，垂直于法向量）
        :param plane: 平面方程(A,B,C,D)
        :return: 法向量n，u轴向量，v轴向量
        """
        A, B, C, D = plane
        n = np.array([A, B, C])
        if abs(np.dot(n, [0, 0, 1])) > 0.999:
            ref = np.array([1, 0, 0])
        else:
            ref = np.array([0, 0, 1])
        u = np.cross(n, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return n, u, v