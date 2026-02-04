import os
import logging
import open3d as o3d
import numpy as np
import pygmsh
import pyvista as pv
import glob

from fit.io import save_plane_params, save_line_params
from fit.k_means import clustering
from fit.neighbor_features import neighbor_features
from fit.fitters import concrete_fit, rebar_fit
from utils.noise_utils import add_gaussian_noise
from rebuild.mesh import build_face, generate_lines_and_points
from rebuild.topology import (
    visualize_cross_lines,
    get_cross_points_by_cross_lines,
    get_cross_lines_by_plane,
)
from geometry import BasePlane

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # ==============0 定义参数==============
    INPUT_FILE = "mock/res/point/prefabricated_laminated_beam.ply"  # 输入点云文件
    OUTPUT_DIR = "output"  # 输出目录
    PCD_NAME = "prefabricated_laminated_beam"  # "prefabricated_column"
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
    CONCRETE_THRESHOLD = 0.001  # 阈值
    CONCRETE_ITERATIONS = 2000  # 迭代次数
    CONCRETE_MIN_INLIERS = 10  # 最小内点数量
    CONCRETE_MAX_N_POINTS = 4  # 最大点数量
    CONCRETE_MAX_PLANE_NUM = 20  # 最大平面数量
    # 钢筋拟合参数
    REBAR_THRESHOLD = 0.01  # 阈值
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
    # pcd = add_gaussian_noise(pcd)
    # 转换为 numpy 数组
    point_data_np = np.asarray(pcd.points)

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
        threshold=REBAR_THRESHOLD,
        iterations=REBAR_ITERATIONS,
        min_inliers=REBAR_MIN_INLIERS,
        max_n_points=REBAR_MAX_N_POINTS,
        linearity_threshold=REBAR_LINEARITY_THRESHOLD,
        random_state=RANDOM_STATE,
    )
    line_count = len(line_fitters)
    logging.info(f"成功拟合 {line_count} 个钢筋线")

    # ==============7 保存混凝土与钢筋拟合结果==============
    logging.info("保存混凝土与钢筋拟合结果...")
    pcd_list = save_plane_params(plane_fitter, OUTPUT_DIR, PCD_NAME)

    # 可视化下采样后的点云
    o3d.visualization.draw_geometries(pcd_list)

    # ==============8 保存钢筋拟合结果==============
    logging.info("保存钢筋拟合结果...")
    pcd_list += save_line_params(line_fitters, OUTPUT_DIR, PCD_NAME)

    # 可视化所有内点
    logging.info("可视化所有内点...")
    o3d.visualization.draw_geometries(pcd_list)

    # ==============9 三维重建==============
    logging.info("三维重建...")
    pcds: list[o3d.geometry.PointCloud] = []
    planes = []
    # 自动遍历所有concrete_plane_params_{i}.npy文件
    plane_param_files = glob.glob(
        f"{OUTPUT_DIR}/{PCD_NAME}/concrete_plane_params_*.npy"
    )
    for i in range(plane_count):
        pcd_file = f"{OUTPUT_DIR}/{PCD_NAME}/concrete_inliers_{i}.ply"
        plane_file = f"{OUTPUT_DIR}/{PCD_NAME}/concrete_plane_params_{i}.npy"
        # 读取点云
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcds.append(pcd)

        # 读取平面参数方程
        plane: BasePlane = BasePlane.read_from_npy(plane_file)
        planes.append(plane)
        logging.info(f"成功加载平面 {i}: 点云文件 {pcd_file}, 参数文件 {plane_file}")

    logging.info(f"成功加载 {len(planes)} 个平面和对应的点云")
    cross_lines = get_cross_lines_by_plane(pcds, planes)
    visualize_cross_lines(cross_lines)
    cross_points, cross_lines = get_cross_points_by_cross_lines(
        cross_lines, expand_list=EXAPND_LIST
    )
    logging.info(f"交线数量: {len(cross_lines)}")
    logging.info(f"交点数量: {len(cross_points)}")
    visualize_cross_lines(cross_lines)

    with pygmsh.geo.Geometry() as geom:
        # 定义实体的顶点和棱
        geom_points, geom_lines = generate_lines_and_points(
            cross_points, cross_lines, geom
        )

        # 由线环构建面
        faces = build_face(planes, cross_lines, geom_lines, geom)

        # 构建闭合面环
        surf_loop = geom.add_surface_loop(faces)

        # 面环围合出三维实体
        volume_3d = geom.add_volume(surf_loop)

        # 生成网格并保存
        mesh = geom.generate_mesh()
        mesh.write(f"{OUTPUT_DIR}/{PCD_NAME}/mesh.vtk", file_format="vtk")

    logging.info(f"三维面包围体网格生成完成，文件：{OUTPUT_DIR}/{PCD_NAME}/mesh.vtk")

    # ==============10 可视化网格==============
    grid = pv.read(f"{OUTPUT_DIR}/{PCD_NAME}/mesh.vtk")
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, color="white")
    plotter.add_axes(line_width=2)
    plotter.show()
