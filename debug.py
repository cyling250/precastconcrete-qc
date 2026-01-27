from split.fitters import visualize_plane_fitter
from split.fitters import close_rebar
from utils.format_utils import save_as_xyzc
from split.k_means import clustering
from split.neighbor_features import neighbor_features
from split.fitters import concrete_fit, rebar_fit
import open3d as o3d
import numpy as np


if __name__ == "__main__":
    TEST_PCD = "mock/res/point/预制叠合梁300x600.ply"
    NEIGHBOR_FEATURES = "output/neibor_features_50mm.npy"
    FEATURE_WEIGHTS = [
        1.0,  # X坐标
        1.0,  # Y坐标
        1.0,  # Z坐标
        5.0,  # linearity（线性度）
        5.0,  # planarity（平面度）
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
        distance=50,
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
    # DEBUG 聚类结果可视化
    # plane_pcd = pcd.select_by_index(np.where(labels == 0)[0])
    # rebar_pcd = pcd.select_by_index(np.where(labels == 1)[0])
    # o3d.visualization.draw_geometries([plane_pcd])
    # o3d.visualization.draw_geometries([rebar_pcd])
    # 4 混凝土与钢筋拟合
    plane_fitter = concrete_fit(
        point_data_np,
        labels,
        NEIGHBOR_FEATURES,
        threshold=0.001,
        iterations=2000,
        min_inliers=10,
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
        threshold=0.01,
        iterations=2000,
        min_inliers=10,
        max_n_points=10,
        linearity_threshold=0.6,
        random_state=42,
    )
    # for line_fitter in line_fitters:
    #     visualize_line_fitter(line_fitter)
    # 钢筋拟合效果再优化
    all_point = plane_fitter.get_inliers()
    # 为混凝土点云进行体素下采样
    show_list = []
    for i in range(len(all_point)):
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(all_point[i])
        # 点云的体积选取合适的体素大小，当点云数量少于500个时，不进行下采样
        if len(all_point[i]) > 1000:
            all_point[i] = pcd_temp.voxel_down_sample(voxel_size=0.03).points
        # 赋予随机颜色
        random_color = np.random.rand(3)  # 生成随机RGB颜色
        pcd_temp.colors = o3d.utility.Vector3dVector(
            np.tile(random_color, (len(all_point[i]), 1))
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

    # 保存为xyzc文件
    save_as_xyzc(all_point, "output/line_fit_inliers.xyzc")
    # for i in range(len(inliers_list)):
    #     # 获取当前直线的内点
    #     current_inliers = inliers_list[i]
    # if len(current_inliers) == 0:
    #     continue

    # # 提取一个点云样本（使用第一个点）
    # sample_point = current_inliers[0]
    # # 获取该直线的方向向量
    # direction = line_equations[i][3:]

    # # 构造垂直于直线的平面（法向量为直线方向向量）
    # # 平面方程: n·(x - p) = 0，其中n是法向量，p是平面上的一点
    # plane_normal = direction

    # # 将内点向平面投影
    # # 投影公式: x_projected = x - [(x - p)·n] * n
    # # 这里p取sample_point
    # inliers_centered = current_inliers - sample_point
    # projections = np.dot(inliers_centered, plane_normal)
    # inliers2d = current_inliers - projections[:, np.newaxis] * plane_normal

    # # 可视化
    # inliers_pcd = o3d.geometry.PointCloud()
    # inliers_pcd.points = o3d.utility.Vector3dVector(inliers2d)
    # # 为每个投影点云设置随机颜色
    # random_color = np.random.rand(3)
    # inliers_pcd.colors = o3d.utility.Vector3dVector(
    #     np.tile(random_color, (len(inliers2d), 1))
    # )
    # o3d.visualization.draw_geometries([inliers_pcd])
