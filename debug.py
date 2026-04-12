import os
import logging
import open3d as o3d
import numpy as np
import pyvista as pv

from fit.io import save_plane_params, save_line_params
from fit.k_means import clustering
from fit.neighbor_features import neighbor_features
from fit.fitters import concrete_fit, rebar_fit
from utils.noise_utils import add_gaussian_noise
from rebuild.rebar_params import (
    get_rebar_param_by_line_fitter,
    get_rebar_param_by_line_fitter_v2,
)
from rebuild.topo_reconstr import concrete_reconstr
from utils.visualization_utils import (
    visualize_point_cloud,
    visualize_clusters,
    visualize_neighbor_features,
    visualize_fitting_results,
)

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    DEBUG = False
    # ==============0 定义参数==============
    # PCD_NAME = "prefabricated_laminated_beam"
    PCD_NAME = "prefabricated_column"
    INPUT_FILE = f"mock/res/point/{PCD_NAME}.ply"  # 输入点云文件
    OUTPUT_DIR = "output"  # 输出目录
    NEIBOR_DISTANCE = 40  # 邻域距离
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
    ]  # 特征权重
    RANDOM_STATE = 42  # 随机种子

    # 混凝土拟合参数
    CONCRETE_THRESHOLD = 0.005  # 阈值
    CONCRETE_ITERATIONS = 2000  # 迭代次数
    CONCRETE_MIN_INLIERS = 10  # 最小内点数量
    CONCRETE_MAX_N_POINTS = 4  # 最大点数量
    CONCRETE_MAX_PLANE_NUM = 6  # 最大平面数量
    # 钢筋拟合参数
    REBAR_ITERATIONS = 2000  # 迭代次数
    REBAR_MIN_INLIERS = 10  # 最小内点数量
    REBAR_MAX_N_POINTS = 10  # 最大点数量
    REBAR_LINEARITY_THRESHOLD = 0.1  # 线性度阈值
    # 三维重建参数
    EXAPND_LIST = [0, 0.01, 0.02, 0.03, 0.04, 0.05]  # 直线向外的扩展列表

    # ==============1 参数准备==============
    if not os.path.exists(f"{OUTPUT_DIR}/{PCD_NAME}"):
        os.makedirs(f"{OUTPUT_DIR}/{PCD_NAME}")
    NEIGHBOR_FEATURES = (
        f"{OUTPUT_DIR}/{PCD_NAME}/neibor_features_{NEIBOR_DISTANCE}mm.npy"
    )

    # ==============2 读取点云文件==============
    logging.info(f"读取点云文件: {INPUT_FILE}")
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    # 添加噪声
    pcd = add_gaussian_noise(pcd, std=0.001)
    # 体素滤波
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    # 转换为 numpy 数组
    point_data_np = np.asarray(pcd.points)
    # 可视化
    visualize_point_cloud(pcd, "原始点云可视化", DEBUG)

    # ==============3 计算邻域特征==============
    logging.info("计算邻域特征...")
    neighbor_features(
        point_data_np,
        NEIGHBOR_FEATURES,
        distance=NEIBOR_DISTANCE,
    )
    # ==============4 点云聚类==============
    logging.info("聚类中...")
    labels = clustering(
        pcd,
        NEIGHBOR_FEATURES,
        output_file=f"{OUTPUT_DIR}/{PCD_NAME}/cluster.ply",
        feature_weights=FEATURE_WEIGHTS,
    )
    # 可视化聚类结果
    visualize_clusters(pcd, labels, DEBUG)
    # 加载邻域特征数据
    visualize_neighbor_features(
        point_data_np,
        NEIGHBOR_FEATURES,
        OUTPUT_DIR,
        PCD_NAME,
        DEBUG,
    )

    # ==============5 混凝土拟合==============
    logging.info("混凝土拟合...")
    plane_fitter = concrete_fit(
        point_data_np,
        labels,
        NEIGHBOR_FEATURES,
        threshold=CONCRETE_THRESHOLD,
        iterations=CONCRETE_ITERATIONS,
        min_inliers=CONCRETE_MIN_INLIERS,
        max_n_points=CONCRETE_MAX_N_POINTS,
        max_plane_num=CONCRETE_MAX_PLANE_NUM,
        debug=DEBUG,
    )
    plane_count = len(plane_fitter.get_plane())
    logging.info(f"成功拟合 {plane_count} 个混凝土平面")

    # ==============6 钢筋拟合==============
    logging.info("钢筋拟合...")
    line_fitters = rebar_fit(
        point_data_np,
        plane_fitter,
        labels,
        NEIGHBOR_FEATURES,
        neighbot_distance=NEIBOR_DISTANCE / 1000,
        iterations=REBAR_ITERATIONS,
        min_inliers=REBAR_MIN_INLIERS,
        max_n_points=REBAR_MAX_N_POINTS,
        linearity_threshold=REBAR_LINEARITY_THRESHOLD,
        concrete_distance_threshold=CONCRETE_THRESHOLD,
        random_state=RANDOM_STATE,
    )
    line_count = len(line_fitters)
    logging.info(f"成功拟合 {line_count} 个钢筋线")

    # ==============7 保存混凝土与钢筋拟合结果==============
    logging.info("保存混凝土与钢筋拟合结果...")
    pcd_list = save_plane_params(plane_fitter, OUTPUT_DIR, PCD_NAME)

    # 可视化下采样后的点云
    visualize_fitting_results(pcd_list, DEBUG)

    # ==============8 保存钢筋拟合结果==============
    logging.info("保存钢筋拟合结果...")
    pcd_list += save_line_params(line_fitters, OUTPUT_DIR, PCD_NAME)

    # 可视化所有内点
    logging.info("可视化所有内点...")
    visualize_fitting_results(pcd_list, DEBUG)

    # ==============9 三维重建==============
    logging.info("混凝土三维重建...")
    concrete_reconstr(OUTPUT_DIR, PCD_NAME, expand_list=EXAPND_LIST)

    logging.info("钢筋重建...")
    rebar_param_list = get_rebar_param_by_line_fitter_v2(line_fitters, 1)

    # ==============10 可视化网格==============
    logging.info("可视化网格...")
    # 可视化混凝土网格
    grid_concrete = pv.read(f"{OUTPUT_DIR}/{PCD_NAME}/mesh-concrete.vtk")
    grid_rebar = pv.read(f"{OUTPUT_DIR}/{PCD_NAME}/mesh-rebar.vtk")
    plotter = pv.Plotter()
    plotter.add_mesh(grid_concrete, show_edges=True, color="red", opacity=0.9)
    plotter.add_mesh(grid_rebar, show_edges=True, color="lightgray", opacity=0.7)
    plotter.add_axes(line_width=2)
    plotter.show()
