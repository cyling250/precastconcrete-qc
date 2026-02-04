from fit.io import save_plane_params, save_line_params
from fit.k_means import clustering
from fit.neighbor_features import neighbor_features
from fit.fitters import concrete_fit, rebar_fit
from utils.noise_utils import add_gaussian_noise
import open3d as o3d
import numpy as np
import os
import logging
from rebuild.mesh import build_face
from rebuild.mesh import generate_lines_and_points
from rebuild.topology import visualize_cross_lines
from rebuild.topology import get_cross_points_by_cross_lines
from rebuild.topology import get_cross_lines_by_plane
import open3d as o3d
import pygmsh
import pyvista as pv
from geometry import BasePlane

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
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
    logging.info(f"读取点云文件: {TEST_PCD}")
    pcd = o3d.io.read_point_cloud(TEST_PCD)
    # 添加噪声
    # pcd = add_gaussian_noise(pcd)
    # 转换为 numpy 数组
    point_data_np = np.asarray(pcd.points)
    # 2 计算邻域特征
    logging.info("计算邻域特征...")
    neighbor_features(
        point_data_np,
        NEIGHBOR_FEATURES,
        distance=40,
    )
    # 3 点云聚类
    logging.info("聚类中...")
    labels = clustering(
        pcd,
        NEIGHBOR_FEATURES,
        output_file=f"output/{PCD_NAME}/cluster.ply",
        feature_weights=FEATURE_WEIGHTS,
    )

    # 4 混凝土拟合
    logging.info("混凝土拟合...")
    plane_fitter = concrete_fit(
        point_data_np,
        labels,
        NEIGHBOR_FEATURES,
        threshold=0.001,
        iterations=2000,
        min_inliers=10,
        max_n_points=4,
        max_plane_num=20,
    )

    # 5 钢筋拟合
    logging.info("钢筋拟合...")
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
    )

    # 6 保存混凝土与钢筋拟合结果
    logging.info("保存混凝土与钢筋拟合结果...")
    pcd_list = save_plane_params(plane_fitter, PCD_NAME)

    # 可视化下采样后的点云
    o3d.visualization.draw_geometries(pcd_list)

    # 保存钢筋拟合结果
    logging.info("保存钢筋拟合结果...")
    pcd_list += save_line_params(line_fitters, PCD_NAME)

    # 可视化所有内点
    logging.info("可视化所有内点...")
    o3d.visualization.draw_geometries(pcd_list)
    # 三维重建
    pcds: list[o3d.geometry.PointCloud] = []
    planes = []
    for i in range(26):
        # 读取点云
        pcd = o3d.io.read_point_cloud(f"output/{PCD_NAME}/concrete_inliers_{i}.ply")
        pcds.append(pcd)

        # 读取平面参数方程
        plane: BasePlane = BasePlane.read_from_npy(
            f"output/{PCD_NAME}/concrete_plane_params_{i}.npy"
        )
        planes.append(plane)
    cross_lines = get_cross_lines_by_plane(pcds, planes)
    cross_points, cross_lines = get_cross_points_by_cross_lines(cross_lines)
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
        mesh.write(f"output/{PCD_NAME}/mesh.vtk", file_format="vtk")

    logging.info(f"三维面包围体网格生成完成，文件：output/{PCD_NAME}/mesh.vtk")
    # 直接读取之前导出的网格可视化
    grid = pv.read(f"output/{PCD_NAME}/mesh.vtk")
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, color="white")
    plotter.add_axes(line_width=2)
    plotter.show()
