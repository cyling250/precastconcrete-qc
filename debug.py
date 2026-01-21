from split.k_means import clustering
from split.neighbor_features import neighbor_features
from split.get_part import get_concrete_part, get_rebar_part
from models.plane_fitter import RanscaPlaneFitter, PlaneModel
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
    point_cloud = np.asarray(pcd.points)
    # 2 计算邻域特征
    neighbor_features(point_cloud, NEIGHBOR_FEATURES, distance=50)
    # 3 点云聚类
    print("聚类中...")
    labels = clustering(
        pcd,
        NEIGHBOR_FEATURES,
        output_file="output/cluster.ply",
        feature_weights=FEATURE_WEIGHTS,
    )
    # 4 混凝土平面拟合
    # 初次聚类后 获取混凝土部分
    concrete_pcd = np.asarray(
        get_concrete_part(point_cloud, labels, NEIGHBOR_FEATURES).points
    )

    # 执行RANSAC
    plane_fitter = RanscaPlaneFitter()
    plane_fitter.fit(concrete_pcd)  # 调用RANSAC平面检测函数
    inliers = plane_fitter.get_inliers()
    plane_equation = plane_fitter.get_plane_equation()
    # inlier_pcd = []
    # for i, temp_inliers in enumerate(inliers):
    #     inlier_pcd.append(
    #         o3d.geometry.PointCloud()
    #     )
    #     inlier_pcd[-1].points = o3d.utility.Vector3dVector(temp_inliers)

    #     # 为每个平面生成随机颜色
    #     random_color = np.random.rand(3)  # 生成随机RGB颜色
    #     inlier_pcd[-1].colors = o3d.utility.Vector3dVector(
    #         np.tile(random_color, (temp_inliers.shape[0], 1))
    #     )
    # o3d.visualization.draw_geometries(inlier_pcd)
    # # 获取外点
    # outliers = plane_fitter.get_outliers()  # 获取外点坐标
    # outlier_pcd = o3d.geometry.PointCloud()
    # outlier_pcd.points = o3d.utility.Vector3dVector(outliers)
    # outlier_pcd.colors = o3d.utility.Vector3dVector(
    #     np.zeros((outliers.shape[0], 3)) + 0.8
    # )  # 黄色
    # # 可视化
    # o3d.visualization.draw_geometries([outlier_pcd])

    # 5 钢筋拟合
    # 5.1 获取初次聚类后的钢筋部分
    rebars_pcd = np.asarray(
        get_rebar_part(point_cloud, labels, NEIGHBOR_FEATURES).points
    )
    # 5.2 在钢筋点云上净化非钢筋点
    # 5.2.1 遍历混凝土平面函数和分割后内点列表
    for i in range(len(inliers)):
        # 计算当前平面与所有点的距离
        plane_model = PlaneModel()
        plane_model.set_parameters(plane_equation[i])
        # 筛选出距离小于阈值的点作为需要删除的部分
        rebars_outliers = rebars_pcd[
            plane_model.calc_inliers(rebars_pcd, plane_fitter._threshold)
        ]
        # 可视化
        rebars_outliers_pcd = o3d.geometry.PointCloud()
        rebars_outliers_pcd.points = o3d.utility.Vector3dVector(rebars_outliers)
        rebars_outliers_pcd.colors = o3d.utility.Vector3dVector(
            np.zeros((rebars_outliers.shape[0], 3)) + 0.8
        )  # 黄色
        concrete_inliers_pcd = o3d.geometry.PointCloud()
        concrete_inliers_pcd.points = o3d.utility.Vector3dVector(inliers[i])
        concrete_inliers_pcd.colors = o3d.utility.Vector3dVector(
            np.zeros((inliers[i].shape[0], 3)) + 0.5
        )  # 黄色
        # 可视化
        o3d.visualization.draw_geometries([rebars_outliers_pcd, concrete_inliers_pcd])
        # # 遍历这些点
        # for j in range(rebars_outliers.shape[0]):
        #     # 如果该点到当前混凝土点云凸包的距离大于阈值
        #     if np.linalg.norm(rebars_outliers[j] - inliers[i][0]) > plane_fitter._threshold:
        #         # 从钢筋点云删除该点
        #         rebars_pcd = np.delete(rebars_pcd, j, axis=0)
