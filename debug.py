from split.k_means import clustering
from split.neighbor_features import neighbor_features
from split.get_part import get_concrete_part
import open3d as o3d
import numpy as np
import random


# RANSAC 平面检测函数
def ransac_plane(xyz, threshold=0.05, iterations=1000):
    inliers = []  # 初始化内点列表
    n_points = len(xyz)  # 获取点云的点数
    i = 0  # 初始化迭代计数器
    while i < iterations:  # 迭代进行RANSAC
        idx_samples = random.sample(range(n_points), 3)  # 随机选择3个点
        pts = xyz[idx_samples]  # 获取这3个点的坐标
        vecA = pts[1] - pts[0]  # 计算向量A
        vecB = pts[2] - pts[0]  # 计算向量B
        normal = np.cross(vecA, vecB)  # 计算法向量
        a, b, c = normal / np.linalg.norm(normal)  # 归一化法向量
        d = -np.sum(normal * pts[1])  # 计算平面方程的d值
        distance = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / np.sqrt(
            a**2 + b**2 + c**2
        )  # 计算所有点到平面的距离
        idx_candidates = np.where(np.abs(distance) <= threshold)[
            0
        ]  # 找到所有符合距离阈值的点（内点）
        if len(idx_candidates) > len(inliers):  # 如果当前内点数量大于之前的最大内点数量
            equation = [a, b, c, d]  # 更新平面方程参数
            inliers = idx_candidates  # 更新内点列表
        i += 1  # 迭代计数器加1
    return equation, inliers  # 返回平面方程参数和内点列表


if __name__ == "__main__":
    TEST_PCD = "mock/res/point/预制叠合梁300x600.ply"
    NEIGHBOR_FEATURES = "output/neibor_features_50mm.npy"
    FEATURE_WEIGHTS = [
        0.6,  # X坐标
        0.6,  # Y坐标
        0.6,  # Z坐标
        3.0,  # linearity（线性度）
        3.0,  # planarity（平面度）
        1.0,  # sphericity（球度）
        1.0,  # omnivariance（各向同性）
        1.0,  # anisotropy（各向异性）
        1.0,  # eigenentropy（特征熵）
        1.0,  # tortuosity（曲折度）
        1.0,  # curvature_change（曲率变化）
    ]
    # 读取点云文件
    pcd = o3d.io.read_point_cloud(TEST_PCD)
    # 转换为 numpy 数组
    point_cloud = np.asarray(pcd.points)
    # 计算邻域特征
    neighbor_features(point_cloud, NEIGHBOR_FEATURES, distance=50)
    # 点云聚类
    labels = clustering(
        pcd,
        NEIGHBOR_FEATURES,
        output_file="output/cluster.ply",
        feature_weights=FEATURE_WEIGHTS,
    )
    # 初次聚类后 获取混凝土部分
    concrete_pcd = np.asarray(
        get_concrete_part(point_cloud, labels, NEIGHBOR_FEATURES).points
    )
    # 混凝土平面拟合
    # 执行RANSAC
    eq, idx_inliers = ransac_plane(concrete_pcd, 0.005, 1000)  # 调用RANSAC平面检测函数
    inliers = concrete_pcd[idx_inliers]  # 获取内点坐标

    # 获取外点
    mask = np.ones(len(concrete_pcd), dtype=bool)  # 创建一个全为True的布尔数组
    mask[idx_inliers] = False  # 将内点位置设置为False
    outliers = concrete_pcd[mask]  # 获取外点坐标

    inlier_pcd = o3d.geometry.PointCloud()
    inlier_pcd.points = o3d.utility.Vector3dVector(inliers)
    inlier_pcd.colors = o3d.utility.Vector3dVector(
        np.zeros((inliers.shape[0], 3)) + 0.5
    )  # 灰色
    outlier_pcd = o3d.geometry.PointCloud()
    outlier_pcd.points = o3d.utility.Vector3dVector(outliers)
    outlier_pcd.colors = o3d.utility.Vector3dVector(
        np.zeros((outliers.shape[0], 3)) + 0.8
    )  # 黄色
    # # 可视化
    o3d.visualization.draw_geometries([inlier_pcd, outlier_pcd])
