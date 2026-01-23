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
