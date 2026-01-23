from split.k_means import clustering
from split.neighbor_features import neighbor_features
from models import RansacLineFitter
from split.fitters import concrete_fit, rebar_fit
import open3d as o3d
import numpy as np


if __name__ == "__main__":
    TEST_PCD = "mock/res/point/预制叠合梁300x600.ply"
    NEIGHBOR_FEATURES = "output/neibor_features_50mm.npy"
    FEATURE_WEIGHTS = [
        0.5,  # X坐标
        0.5,  # Y坐标
        0.5,  # Z坐标
        3.0,  # linearity（线性度）
        3.0,  # planarity（平面度）
        1.0,  # sphericity（球度）
        1.0,  # omnivariance（各向同性）
        1.0,  # anisotropy（各向异性）
        1.0,  # eigenentropy（特征熵）
        1.0,  # tortuosity（曲折度）
        1.0,  # curvature_change（曲率变化）
    ]

    # 1 读取点云文件
    pcd = o3d.io.read_point_cloud(TEST_PCD)
    # 转换为 numpy 数组
    point_data_np = np.asarray(pcd.points)
    # 2 计算邻域特征
    neighbor_features(point_data_np, NEIGHBOR_FEATURES, distance=50)
    # 3 点云聚类
    print("聚类中...")
    labels = clustering(
        pcd,
        NEIGHBOR_FEATURES,
        output_file="output/cluster.ply",
        feature_weights=FEATURE_WEIGHTS,
    )
    # 4 混凝土与钢筋拟合
    # 4.1 初次聚类后 获取混凝土部分
    plane_fitter = concrete_fit(point_data_np, labels, NEIGHBOR_FEATURES)

    # 5 钢筋拟合
    rebar_fit(
        point_data_np,
        plane_fitter,
        labels,
        NEIGHBOR_FEATURES,
    )
