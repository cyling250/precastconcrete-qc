import numpy as np


class BasePlane(object):
    def __init__(self, params: np.ndarray):
        # 平面模型参数（平面上任意一点 cx, cy, cx） + （平面法向量 nx, ny, nz）
        # 如果形状是(4,)则是ABCD，如果形状是(6,)则是cx,cy,cx,nx,ny,nz
        # 在生成时，使用UUID作为平面的唯一标识符
        import uuid

        self._id = uuid.uuid4()
        self._parameters: np.ndarray | None = None
        if type(params) is not np.ndarray:
            raise AttributeError("平面模型参数必须是numpy数组")
        else:
            if params.shape[0] == 4:
                # 转换为cx,cy,cx,nx,ny,nz格式
                c = (params[0:3] + params[3:6]) / 2
                n = params[3:6] - params[0:3]
                self._parameters: np.ndarray | None = np.hstack((c, n))
            elif params.shape[0] == 6:
                self._parameters: np.ndarray | None = params
            else:
                raise ValueError("平面模型参数必须是4或6维数组")
        # 法向量归一化
        self._parameters[3:6] = self._parameters[3:6] / np.linalg.norm(
            self._parameters[3:6]
        )

    def get_standard(self):
        """获取平面的标准形式参数Ax+By+Cz+D=0

        Returns:
            np.ndarray: 标准形式参数，形状为 (4,)
        """
        c = self._parameters[0:3]
        n = self._parameters[3:6]
        d = -np.dot(c, n)
        return np.hstack((n, d))

    @staticmethod
    def read_from_npy(npy_path: str):
        """从npy文件读取平面模型参数

        Args:
            npy_path (str): npy文件路径
        """
        params = np.load(npy_path)
        return BasePlane(params)

    def set_parameters(self, params: np.ndarray):
        """设置平面模型参数

        Args:
            params (np.ndarray): 平面模型参数，形状为 (4,) 或 (6,)
        """
        self._parameters = params

    def get_parameters(self):
        """获取平面模型参数

        Returns:
            np.ndarray: 平面模型参数，形状为 (6,)
        """
        return self._parameters

    def __eq__(self, other) -> bool:
        """判断两个平面是否是同一个平面"""
        return self._id == other._id

    def __hash__(self):
        return hash(self._id)


def calc_inliers(
    plane: BasePlane, points: np.ndarray, dst_threshold: float
) -> np.ndarray:
    """根据点到平面的距离计算内点和外点

    Args:
        points (np.ndarray): 输入点云，形状为 (N, 3)
        dst_threshold (float): 内点距离阈值

    Returns:
        np.ndarray: 内点索引，形状为 (M,)
    """
    if plane._parameters is None:
        raise ValueError("平面模型参数未初始化")
    c = plane._parameters[0:3]
    n = plane._parameters[3:6]
    dst = abs(np.dot(points - c, n))
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
        from .line_model import SVD

        _, _, c, n = SVD(pts)

    params = np.hstack((c.reshape(1, -1), n.reshape(1, -1)))[0, :]
    return BasePlane(params)


from .line_model import CrossLine


def project2plane(plane: BasePlane, points: np.ndarray) -> np.ndarray:
    """将所有点向平面投影，返回投影点云

    Args:
        plane (BasePlane): 平面模型
        points (np.ndarray): 输入点云，形状为 (N, 3)

    Returns:
        np.ndarray: 投影点云，形状为 (N, 3)
    """
    c = plane._parameters[0:3]
    n = plane._parameters[3:6]
    proj_points = points - np.dot(points - c, n.reshape(-1, 1)) * n.reshape(1, -1)
    return proj_points


def intersect_plane(plane1: BasePlane, plane2: BasePlane) -> CrossLine | None:
    """计算当前平面与另一个平面的交线

    Args:
        plane1 (BasePlane): 第一个平面模型
        plane2 (BasePlane): 第二个平面模型

    Returns:
        CrossLine | None: 两个平面的交线模型
    """
    # 获取两个平面的参数
    c1 = plane1._parameters[0:3]  # 平面1上的点
    n1 = plane1._parameters[3:6]  # 平面1的法向量
    c2 = plane2._parameters[0:3]  # 平面2上的点
    n2 = plane2._parameters[3:6]  # 平面2的法向量

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

    return CrossLine(line_params, plane1, plane2)
