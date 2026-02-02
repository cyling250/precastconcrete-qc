from fit.k_means import clustering
from fit.neighbor_features import neighbor_features
from fit.fitters import concrete_fit, rebar_fit
from utils.noise_utils import add_gaussian_noise
import open3d as o3d
import numpy as np
import os


if __name__ == "__main__":
    debug = False
    PCD_NAME = "prefabricated_laminated_beam"
    # PCD_NAME = "prefabricated_column"
    NEIBOR_DISTANCE = 40
    FEATURE_WEIGHTS = [
        1,  # X坐标
        1,  # Y坐标
        1,  # Z坐标
        5.0,  # linearity（线性度）
        5.0,  # planarity（平面度）
        1.0,  # sphericity（球度）
        1.0,  # omnivariance（各向同性）
        1.0,  # anisotropy（各向异性）
        1.0,  # eigenentropy（特征熵）
        1.0,  # tortuosity（曲折度）
        1.0,  # curvature_change（曲率变化）
    ]
    if not os.path.exists(f"output/{PCD_NAME}"):
        os.makedirs(f"output/{PCD_NAME}")
    TEST_PCD = f"mock/res/point/{PCD_NAME}.ply"
    NEIGHBOR_FEATURES = f"output/{PCD_NAME}/neibor_features_{NEIBOR_DISTANCE}mm.npy"
    # 1 读取点云文件
    pcd = o3d.io.read_point_cloud(TEST_PCD)
    # 添加噪声
    # pcd = add_gaussian_noise(pcd)
    # 转换为 numpy 数组
    point_data_np = np.asarray(pcd.points)
    # DEBUG 可视化
    if debug:
        o3d.visualization.draw_geometries([pcd])
    # 2 计算邻域特征
    neighbor_features(
        point_data_np,
        NEIGHBOR_FEATURES,
        distance=40,
        is_visualize=debug,
    )
    # 3 点云聚类
    print("聚类中...")
    labels = clustering(
        pcd,
        NEIGHBOR_FEATURES,
        output_file=f"output/{PCD_NAME}/cluster.ply",
        feature_weights=FEATURE_WEIGHTS,
    )
    # DEBUG 聚类结果可视化
    if debug:
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
        min_inliers=10,
        max_n_points=4,
        max_plane_num=20,
        debug=debug,
    )
    # 5 钢筋拟合
    line_fitters = rebar_fit(
        point_data_np,
        plane_fitter,
        labels,
        NEIGHBOR_FEATURES,
        threshold=0.01,
        iterations=2000,
        min_inliers=10,
        max_n_points=10,
        linearity_threshold=0.1,
        random_state=42,
        debug=debug,
    )

    # 6 保存数据
    # 6.1 为Point2CAD构造数据格式
    all_point = plane_fitter.get_inliers()
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
        # 保存平面点云
        o3d.io.write_point_cloud(
            f"output/{PCD_NAME}/concrete_inliers_{i}.ply", pcd_temp
        )
        # 保存平面公式
        plane_params = plane_fitter.get_plane()[i].get_parameters()
        np.save(f"output/{PCD_NAME}/concrete_plane_params_{i}.npy", plane_params)
        show_list.append(pcd_temp)

    # 可视化下采样后的点云
    o3d.visualization.draw_geometries(show_list)

    # 保存钢筋拟合结果
    i = 0
    for line_fitter in line_fitters:
        i += 1
        inliers_list = line_fitter.get_inliers()
        # 保存点云数据为ply
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(inliers_list[0])
        o3d.io.write_point_cloud(f"output/{PCD_NAME}/rebar_inliers_{i}.ply", pcd_temp)
        # 保存直线公式
        line_params = line_fitter.get_line_equation()
        np.save(f"output/{PCD_NAME}/rebar_line_params_{i}.npy", line_params)
        # 添加原始内点
        all_point.append([i for inliers in inliers_list for i in inliers])

# 可视化所有内点
pcd_temp = o3d.geometry.PointCloud()
pcd_temp.points = o3d.utility.Vector3dVector(np.concatenate(all_point))
o3d.visualization.draw_geometries([pcd_temp])
