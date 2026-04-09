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
from rebuild.concrete_topology import (
    visualize_cross_lines,
    get_cross_points_by_cross_lines,
    get_cross_lines_by_plane,
)
from rebuild.rebar_topology import (
    get_rebar_param_by_line_fitter,
    get_rebar_param_by_line_fitter_v2,
)
from geometry import BasePlane, SegmentLine

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    DEBUG = True
    # ==============0 定义参数==============
    PCD_NAME = "prefabricated_laminated_beam"
    # PCD_NAME = "prefabricated_column"
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
    # 将点云设置为灰色
    pcd.paint_uniform_color([157 / 255, 158 / 255, 163 / 255])  # RGB: 灰色
    o3d.visualization.draw_geometries([pcd])

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
    # ============ DEBUG: 可视化聚类结果 ==============
    if DEBUG:
        for label in np.unique(labels):
            pcd_label = o3d.geometry.PointCloud()
            pcd_label.points = o3d.utility.Vector3dVector(
                np.array(pcd.points)[labels == label]
            )
            # pcd_label.colors = o3d.utility.Vector3dVector(
            #     np.tile(
            #         [130 / 255, 178 / 255, 154 / 255],
            #         (np.array(pcd_label.points).shape[0], 1),
            #     )
            # )
            pcd_label.colors = o3d.utility.Vector3dVector(
                np.tile(
                    [223 / 255, 122 / 255, 94 / 255],
                    (np.array(pcd_label.points).shape[0], 1),
                )
            )
            o3d.visualization.draw_geometries([pcd_label])
            # 可视化邻域线性度

    # ============== END DEBUG ===============

    # ============== 可视化所有邻域特征 ===============
    if DEBUG:
        logging.info("可视化所有邻域特征...")
        # 加载邻域特征数据
        features_dict = np.load(NEIGHBOR_FEATURES, allow_pickle=True).item()

        # 定义所有要可视化的特征
        feature_list = [
            "linearity",  # 线性度
            "planarity",  # 平面度
            "sphericity",  # 球度
            "omnivariance",  # 各向同性
            "anisotropy",  # 各向异性
            "eigenentropy",  # 特征熵
            "tortuosity",  # 曲折度
            "curvature_change",  # 曲率变化
        ]

        # 为每个特征创建可视化
        for feature_name in feature_list:
            feature_data = features_dict.get(feature_name)
            if feature_data is not None:
                logging.info(f"可视化特征: {feature_name}")

                # 创建点云对象
                feature_pcd = o3d.geometry.PointCloud()
                feature_pcd.points = o3d.utility.Vector3dVector(point_data_np)

                # 归一化特征值到 [0, 1] 范围
                feature_normalized = (feature_data - np.min(feature_data)) / (
                    np.max(feature_data) - np.min(feature_data) + 1e-8
                )

                # 使用颜色映射：低值（蓝色）到高值（红色）
                colors = np.zeros((len(feature_normalized), 3))

                # 创建颜色映射：蓝色（低值）-> 绿色（中等值）-> 红色（高值）
                for i, value in enumerate(feature_normalized):
                    if value < 0.5:
                        # 蓝色到绿色
                        colors[i, 0] = 0  # R
                        colors[i, 1] = value * 2  # G
                        colors[i, 2] = 1 - value * 2  # B
                    else:
                        # 绿色到红色
                        colors[i, 0] = (value - 0.5) * 2  # R
                        colors[i, 1] = 1 - (value - 0.5) * 2  # G
                        colors[i, 2] = 0  # B

                feature_pcd.colors = o3d.utility.Vector3dVector(colors)

                # 可视化
                window_title = f"邻域{feature_name}可视化"
                o3d.visualization.draw_geometries(
                    [feature_pcd], window_name=window_title, width=1024, height=768
                )

                # 打印统计信息
                logging.info(f"{feature_name}统计:")
                logging.info(f"  最小值: {np.min(feature_data):.6f}")
                logging.info(f"  最大值: {np.max(feature_data):.6f}")
                logging.info(f"  平均值: {np.mean(feature_data):.6f}")
                logging.info(f"  中位数: {np.median(feature_data):.6f}")
                logging.info(f"  标准差: {np.std(feature_data):.6f}")

                # 保存可视化结果
                output_file = (
                    f"{OUTPUT_DIR}/{PCD_NAME}/{feature_name}_visualization.ply"
                )
                o3d.io.write_point_cloud(output_file, feature_pcd)
                logging.info(f"{feature_name}可视化结果已保存到: {output_file}")

                # 保存特征数据为文本文件
                stats_file = f"{OUTPUT_DIR}/{PCD_NAME}/{feature_name}_stats.txt"
                with open(stats_file, "w", encoding="utf-8") as f:
                    f.write(f"{feature_name}特征统计:\n")
                    f.write(f"最小值: {np.min(feature_data):.6f}\n")
                    f.write(f"最大值: {np.max(feature_data):.6f}\n")
                    f.write(f"平均值: {np.mean(feature_data):.6f}\n")
                    f.write(f"中位数: {np.median(feature_data):.6f}\n")
                    f.write(f"标准差: {np.std(feature_data):.6f}\n")
                    f.write(f"数据点数: {len(feature_data)}\n")
                logging.info(f"{feature_name}统计信息已保存到: {stats_file}")
            else:
                logging.warning(f"邻域特征文件中未找到{feature_name}数据")

        # 创建特征对比可视化（将所有特征显示在一个网格中）
        logging.info("创建特征对比可视化...")
        try:
            # 创建一个包含所有特征可视化的网格
            feature_pcds = []
            feature_names_display = []

            for feature_name in feature_list[:4]:  # 只显示前4个特征以避免窗口过多
                feature_data = features_dict.get(feature_name)
                if feature_data is not None:
                    feature_pcd = o3d.geometry.PointCloud()
                    feature_pcd.points = o3d.utility.Vector3dVector(point_data_np)

                    # 归一化特征值
                    feature_normalized = (feature_data - np.min(feature_data)) / (
                        np.max(feature_data) - np.min(feature_data) + 1e-8
                    )

                    # 使用简单的热图颜色映射
                    colors = np.zeros((len(feature_normalized), 3))
                    colors[:, 0] = feature_normalized  # 红色通道
                    colors[:, 1] = 0  # 绿色通道
                    colors[:, 2] = 1 - feature_normalized  # 蓝色通道

                    feature_pcd.colors = o3d.utility.Vector3dVector(colors)
                    feature_pcds.append(feature_pcd)
                    feature_names_display.append(feature_name)

            # 可视化多个特征
            if feature_pcds:
                # 创建一个自定义可视化器
                vis = o3d.visualization.Visualizer()
                vis.create_window(
                    window_name="邻域特征对比可视化", width=1600, height=900
                )

                # 添加所有点云
                for i, pcd in enumerate(feature_pcds):
                    vis.add_geometry(pcd)

                # 设置视图
                vis.get_render_option().point_size = 2.0
                vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

                # 运行可视化
                vis.run()
                vis.destroy_window()

                logging.info("特征对比可视化完成")
        except Exception as e:
            logging.error(f"创建特征对比可视化失败: {e}")

    # ============== END 可视化所有邻域特征 ===============

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
    o3d.visualization.draw_geometries(pcd_list)

    # ==============8 保存钢筋拟合结果==============
    logging.info("保存钢筋拟合结果...")
    pcd_list += save_line_params(line_fitters, OUTPUT_DIR, PCD_NAME)

    # 可视化所有内点
    logging.info("可视化所有内点...")
    o3d.visualization.draw_geometries(pcd_list)

    # ==============9 三维重建==============
    logging.info("混凝土三维重建...")
    concrete_pcds: list[o3d.geometry.PointCloud] = []
    concrete_planes: list[BasePlane] = []
    # 自动遍历所有concrete_plane_params_{i}.npy文件
    concrete_plane_param_files = glob.glob(
        f"{OUTPUT_DIR}/{PCD_NAME}/concrete_plane_params_*.npy"
    )
    for i in range(plane_count):
        tmp_pcd_file = f"{OUTPUT_DIR}/{PCD_NAME}/concrete_inliers_{i}.ply"
        tmp_plane_file = f"{OUTPUT_DIR}/{PCD_NAME}/concrete_plane_params_{i}.npy"
        # 读取点云
        pcd = o3d.io.read_point_cloud(tmp_pcd_file)
        concrete_pcds.append(pcd)

        # 读取平面参数方程
        plane: BasePlane = BasePlane.read_from_npy(tmp_plane_file)
        concrete_planes.append(plane)
        logging.info(
            f"成功加载混凝土平面 {i}: 点云文件 {tmp_pcd_file}, 参数文件 {tmp_plane_file}"
        )
    logging.info(f"成功加载 {len(concrete_planes)} 个混凝土平面和对应的点云")

    logging.info("计算交线...")
    concrete_cross_lines = get_cross_lines_by_plane(concrete_pcds, concrete_planes)
    visualize_cross_lines(concrete_cross_lines)
    logging.info(f"交线数量: {len(concrete_cross_lines)}")

    logging.info("计算线段交点...")
    concrete_cross_points, concrete_segment_lines = get_cross_points_by_cross_lines(
        concrete_cross_lines, expand_list=EXAPND_LIST
    )
    logging.info(f"交点数量: {len(concrete_cross_points)}")

    visualize_cross_lines(concrete_segment_lines)

    logging.info("钢筋重建...")
    rebar_param_list = get_rebar_param_by_line_fitter_v2(line_fitters, 1)

    with pygmsh.occ.Geometry() as geom:
        # 定义实体的顶点和棱
        geom_points, geom_lines = generate_lines_and_points(
            concrete_cross_points, concrete_segment_lines, geom
        )

        # 由线环构建面
        faces = build_face(concrete_planes, concrete_cross_lines, geom_lines, geom)

        # 构建闭合面环
        surf_loop = geom.add_surface_loop(faces)

        # 面环围合出三维实体
        volume_3d = geom.add_volume(surf_loop)

        # 生成网格并保存
        mesh = geom.generate_mesh()
        mesh.write(f"{OUTPUT_DIR}/{PCD_NAME}/mesh-concrete.vtk", file_format="vtk")
        logging.info(
            f"三维面包围体网格生成完成，文件：{OUTPUT_DIR}/{PCD_NAME}/mesh-concrete.vtk"
        )

    with pygmsh.occ.Geometry() as geom:
        # 按照三维圆柱添加钢筋
        for rebar_params in rebar_param_list:
            tmp_rebar_3d = []
            for rebar_param in rebar_params:
                segment_line: SegmentLine = rebar_param[0]
                radius: float = rebar_param[1]

                # 检查钢筋参数是否有效
                if radius <= 0 or segment_line is None:
                    logging.warning(
                        f"跳过无效的钢筋参数: 半径={radius}, segment_line={segment_line}"
                    )
                    continue

                # 计算钢筋长度和方向
                start_point = segment_line.link_points[0].location
                end_point = segment_line.link_points[1].location
                direction = end_point - start_point

                # 检查方向向量是否有效
                if np.linalg.norm(direction) < 1e-6:
                    logging.warning(f"跳过长度过小的钢筋: {np.linalg.norm(direction)}")
                    continue

                # 添加圆柱体
                try:
                    cylinder = geom.add_cylinder(
                        start_point,
                        direction,
                        radius=radius,
                        mesh_size=radius,
                    )
                    tmp_rebar_3d.append(cylinder)
                    logging.debug(f"成功添加钢筋圆柱体，半径={radius:.4f}")
                except Exception as e:
                    logging.error(f"添加钢筋圆柱体失败: {e}")
                    continue

            # 如果有有效的钢筋实体，进行布尔并集操作
            if len(tmp_rebar_3d) > 1:
                # geom.boolean_union(entities=tmp_rebar_3d)
                logging.info(f"成功合并{len(tmp_rebar_3d)}个钢筋实体")
            elif len(tmp_rebar_3d) == 1:
                logging.info("只有一个钢筋实体，无需并集操作")
            else:
                logging.warning("没有有效的钢筋实体可生成网格")

        # 生成网格并保存
        mesh = geom.generate_mesh()
        mesh.write(f"{OUTPUT_DIR}/{PCD_NAME}/mesh-rebar.vtk", file_format="vtk")
        logging.info(f"钢筋网格生成完成，文件：{OUTPUT_DIR}/{PCD_NAME}/mesh-rebar.vtk")

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
