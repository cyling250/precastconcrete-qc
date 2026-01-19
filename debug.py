from split.k_means import clustering
from split.neighbor_features import neighbor_features
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    TEST_PCD = "mock/res/point/预制叠合梁300x600.ply"
    NNEIGHBOR_FEATURES = "output/neibor_features_50mm.npy"
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
    neighbor_features(point_cloud, NNEIGHBOR_FEATURES, distance=50)
    clustering(
        pcd,
        NNEIGHBOR_FEATURES,
        output_file="output/cluster.ply",
        feature_weights=FEATURE_WEIGHTS,
    )
    # 可视化聚类结果
    pcd = o3d.io.read_point_cloud("output/cluster.ply")
    o3d.visualization.draw_geometries([pcd])
