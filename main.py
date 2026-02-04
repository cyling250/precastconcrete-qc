import os
import logging
import argparse
import json
import open3d as o3d
import numpy as np
import pygmsh
import glob

from fit.io import save_plane_params, save_line_params
from fit.k_means import clustering
from fit.neighbor_features import neighbor_features
from fit.fitters import concrete_fit, rebar_fit
from utils.noise_utils import add_gaussian_noise
from rebuild.mesh import build_face, generate_lines_and_points
from rebuild.topology import (
    get_cross_points_by_cross_lines,
    get_cross_lines_by_plane,
)
from geometry import BasePlane

logging.basicConfig(level=logging.INFO)


def load_feature_weights(weights_file):
    """从JSON文件加载特征权重"""
    if not os.path.exists(weights_file):
        logging.warning(f"特征权重文件 {weights_file} 不存在，使用默认权重")
        return [
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

    with open(weights_file, "r", encoding="utf-8") as f:
        weights_data = json.load(f)

    if isinstance(weights_data, list) and len(weights_data) == 11:
        return weights_data
    else:
        logging.warning(f"特征权重文件格式不正确，使用默认权重")
        return [1, 1, 1, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="预制混凝土构件点云处理与三维重建")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="mock/res/point/prefabricated_laminated_beam.ply",
        help="输入点云文件路径",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="output", help="输出目录"
    )
    parser.add_argument(
        "--pcd-name",
        "-n",
        type=str,
        default="prefabricated_laminated_beam",
        help="点云名称，用于创建输出子目录",
    )
    parser.add_argument(
        "--neighbor-distance", "-d", type=int, default=40, help="邻域距离（毫米）"
    )
    parser.add_argument(
        "--feature-weights",
        "-w",
        type=str,
        default="config/feature_weights.json",
        help="特征权重JSON文件路径",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--concrete-threshold", type=float, default=0.001, help="混凝土拟合阈值"
    )
    parser.add_argument(
        "--concrete-iterations", type=int, default=2000, help="混凝土拟合迭代次数"
    )
    parser.add_argument(
        "--concrete-min-inliers", type=int, default=10, help="混凝土拟合最小内点数量"
    )
    parser.add_argument(
        "--concrete-max-points", type=int, default=4, help="混凝土拟合最大点数量"
    )
    parser.add_argument(
        "--concrete-max-planes", type=int, default=20, help="混凝土拟合最大平面数量"
    )
    parser.add_argument(
        "--rebar-threshold", type=float, default=0.01, help="钢筋拟合阈值"
    )
    parser.add_argument(
        "--rebar-iterations", type=int, default=2000, help="钢筋拟合迭代次数"
    )
    parser.add_argument(
        "--rebar-min-inliers", type=int, default=10, help="钢筋拟合最小内点数量"
    )
    parser.add_argument(
        "--rebar-max-points", type=int, default=10, help="钢筋拟合最大点数量"
    )
    parser.add_argument(
        "--rebar-linearity-threshold",
        type=float,
        default=0.1,
        help="钢筋拟合线性度阈值",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # ==============0 定义参数==============
    INPUT_FILE = args.input
    OUTPUT_DIR = args.output_dir
    PCD_NAME = args.pcd_name
    NEIBOR_DISTANCE = args.neighbor_distance
    FEATURE_WEIGHTS = load_feature_weights(args.feature_weights)
    RANDOM_STATE = args.random_state
    # 混凝土拟合参数
    CONCRETE_THRESHOLD = args.concrete_threshold
    CONCRETE_ITERATIONS = args.concrete_iterations
    CONCRETE_MIN_INLIERS = args.concrete_min_inliers
    CONCRETE_MAX_N_POINTS = args.concrete_max_points
    CONCRETE_MAX_PLANE_NUM = args.concrete_max_planes
    # 钢筋拟合参数
    REBAR_THRESHOLD = args.rebar_threshold
    REBAR_ITERATIONS = args.rebar_iterations
    REBAR_MIN_INLIERS = args.rebar_min_inliers
    REBAR_MAX_N_POINTS = args.rebar_max_points
    REBAR_LINEARITY_THRESHOLD = args.rebar_linearity_threshold
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

    # ==============8 保存钢筋拟合结果==============
    logging.info("保存钢筋拟合结果...")
    pcd_list += save_line_params(line_fitters, OUTPUT_DIR, PCD_NAME)

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
    cross_points, cross_lines = get_cross_points_by_cross_lines(
        cross_lines, expand_list=EXAPND_LIST
    )
    logging.info(f"交线数量: {len(cross_lines)}")
    logging.info(f"交点数量: {len(cross_points)}")

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
