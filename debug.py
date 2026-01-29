from split.fitters import visualize_plane_fitter
from split.fitters import close_rebar
from utils.format_utils import save_as_xyzc
from split.k_means import clustering
from split.neighbor_features import neighbor_features
from split.fitters import concrete_fit, rebar_fit
import open3d as o3d
import numpy as np


if __name__ == "__main__":
    TEST_PCD = "mock/res/point/预制柱.ply"
    NEIGHBOR_FEATURES = "output/neibor_features_40mm.npy"
    FEATURE_WEIGHTS = [
        1,  # X坐标
        1,  # Y坐标
        1,  # Z坐标
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
    # 可视化
    o3d.visualization.draw_geometries([pcd])
    # 2 计算邻域特征
    neighbor_features(
        point_data_np,
        NEIGHBOR_FEATURES,
        distance=40,
        is_visualize=False,
    )
    # 3 点云聚类
    print("聚类中...")
    labels = clustering(
        pcd,
        NEIGHBOR_FEATURES,
        output_file="output/cluster.ply",
        feature_weights=FEATURE_WEIGHTS,
    )
    # # DEBUG 聚类结果可视化
    plane_pcd = pcd.select_by_index(np.where(labels == 0)[0])
    rebar_pcd = pcd.select_by_index(np.where(labels == 1)[0])
    o3d.visualization.draw_geometries([plane_pcd])
    o3d.visualization.draw_geometries([rebar_pcd])
    # 4 混凝土与钢筋拟合
    plane_fitter = concrete_fit(
        point_data_np,
        labels,
        NEIGHBOR_FEATURES,
        threshold=0.001,
        iterations=2000,
        min_inliers=100,
        max_n_points=5,
        max_plane_num=20,
    )
    visualize_plane_fitter(plane_fitter)
    # 5 钢筋拟合
    line_fitters = rebar_fit(
        point_data_np,
        plane_fitter,
        labels,
        NEIGHBOR_FEATURES,
        threshold=0.05,
        iterations=2000,
        min_inliers=10,
        max_n_points=10,
        linearity_threshold=0.6,
        random_state=42,
    )
    # 保存数据
    all_point = plane_fitter.get_inliers()
    # 为混凝土点云进行体素下采样
    show_list = []
    for i in range(len(all_point)):
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(all_point[i])

        # 点云的体积选取合适的体素大小，当点云数量少于500个时，不进行下采样
        if len(all_point[i]) > 500:
            pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.03)
            all_point[i] = np.asarray(pcd_temp.points)

        # 离群点删除
        cl, ind = pcd_temp.remove_radius_outlier(nb_points=1, radius=0.04)
        pcd_temp.points = o3d.utility.Vector3dVector(np.asarray(pcd_temp.points)[ind])
        # 为每个点云生成不同的随机颜色
        random_color = np.random.rand(3)  # 生成随机RGB颜色
        current_points = np.asarray(pcd_temp.points)
        pcd_temp.colors = o3d.utility.Vector3dVector(
            np.tile(random_color, (len(current_points), 1))
        )
        show_list.append(pcd_temp)
    # 可视化下采样后的点云
    o3d.visualization.draw_geometries(show_list)
    save_as_xyzc(all_point, "output/concrete_inliers.xyzc")

    for line_fitter in line_fitters:
        inliers_list = line_fitter.get_inliers()
        # # 给钢筋封口
        # cap_points = close_rebar(line_fitter)
        # # 添加封口点
        # all_point.extend(cap_points)
        # 添加原始内点
        all_point.append([i for inliers in inliers_list for i in inliers])
    save_as_xyzc(all_point, "output/all_inliers.xyzc")
    show_list = []
    for i in range(len(all_point)):
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(all_point[i])
        # 为每个点云生成不同的随机颜色
        random_color = np.random.rand(3)  # 生成随机RGB颜色
        current_points = np.asarray(pcd_temp.points)
        pcd_temp.colors = o3d.utility.Vector3dVector(
            np.tile(random_color, (len(current_points), 1))
        )
        show_list.append(pcd_temp)
    # 可视化下采样后的点云
    o3d.visualization.draw_geometries(show_list)
