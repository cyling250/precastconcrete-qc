import numpy as np


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


def project_points_to_plane(
    points: np.ndarray, P: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """将所有点向平面投影，返回投影点云

    Args:
        points (np.ndarray): 输入点云，形状为 (N, 3)
        P (np.ndarray): 平面上的点，形状为 (3,)
        n (np.ndarray): 平面的法向量，形状为 (3,)

    Returns:
        np.ndarray: 投影点云，形状为 (N, 3)
    """
    c = P
    n = n / np.linalg.norm(n)
    proj_points = points - np.dot(points - c, n.reshape(-1, 1)) * n.reshape(1, -1)
    return proj_points


def project_points_to_line(points, P, v):
    """
    将3D点投影到直线上。

    参数:
    points (np.ndarray): 输入的3D点，形状为 (3,) 或 (N, 3)。
    P (np.ndarray): 直线上的点，形状为 (3,)。
    v (np.ndarray): 直线的方向向量，形状为 (3,)。

    返回:
    np.ndarray: 投影到直线上的点，形状与输入点相同。
    """

    # 输入类型/维度校验
    points = np.asarray(points, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if P.shape != (3,) or v.shape != (3,):
        raise ValueError("直线的定点和方向向量必须是3维np数组（shape=(3,)）")
    if (
        points.ndim not in (1, 2)
        or (points.ndim == 1 and points.shape != (3,))
        or (points.ndim == 2 and points.shape[1] != 3)
    ):
        raise ValueError("待投影点必须是shape=(3,)的单个点或shape=(N,3)的批量点")

    # 方向向量非零判断
    if np.linalg.norm(v) < 1e-8:
        raise ValueError("直线方向向量不能为零向量")

    # 扩展维度，适配批量点的向量化计算（避免for循环）
    # 若points是单个点(3,)→(1,3)，批量点(N,3)→保持不变
    points_exp = points[np.newaxis, :] if points.ndim == 1 else points
    P_exp = P[np.newaxis, :]  # (1,3)
    v_exp = v[np.newaxis, :]  # (1,3)

    # 核心计算：向量AP = P - A
    vec_ap = points_exp - P_exp  # (1,3) 或 (N,3)
    # 投影系数t = (AP · u) / |u|² （点积/模长平方）
    dot_product = np.sum(vec_ap * v_exp, axis=1, keepdims=True)  # (1,1)或(N,1)
    v_norm_sq = np.sum(v_exp**2, axis=1, keepdims=True)  # (1,1)，|u|²
    t = dot_product / v_norm_sq

    # 计算投影点 P' = A + t*u
    proj_points = P_exp + t * v_exp  # (1,3)或(N,3)

    # 还原维度：若输入是单个点，输出(3,)；批量点输出(N,3)
    if points.ndim == 1:
        proj_points = proj_points.squeeze(0)

    return proj_points
