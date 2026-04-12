"""
可视化工具模块
包含点云、聚类和邻域特征的可视化函数
"""

import numpy as np
import open3d as o3d
import logging


def visualize_point_cloud(
    pcd: o3d.geometry.PointCloud, window_name: str = "点云可视化", debug: bool = False
):
    """
    可视化点云

    Args:
        pcd (o3d.geometry.PointCloud): 点云对象
        window_name (str): 窗口名称
        debug (bool): 是否启用调试模式
    """
    if not debug:
        return
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def visualize_clusters(
    pcd: o3d.geometry.PointCloud, labels: np.ndarray, debug: bool = False
):
    """
    可视化聚类结果

    Args:
        pcd (o3d.geometry.PointCloud): 原始点云
        labels (np.ndarray): 聚类标签
        debug (bool): 是否启用调试模式
    """
    if not debug:
        return

    for label in np.unique(labels):
        pcd_label = o3d.geometry.PointCloud()
        pcd_label.points = o3d.utility.Vector3dVector(
            np.array(pcd.points)[labels == label]
        )
        # 设置颜色
        pcd_label.colors = o3d.utility.Vector3dVector(
            np.tile(
                [223 / 255, 122 / 255, 94 / 255],
                (np.array(pcd_label.points).shape[0], 1),
            )
        )
        o3d.visualization.draw_geometries([pcd_label], window_name=f"聚类 {label}")


def visualize_neighbor_features(
    point_data_np: np.ndarray,
    neighbor_features_file: str,
    output_dir: str,
    pcd_name: str,
    debug: bool = False,
):
    """
    可视化所有邻域特征

    Args:
        point_data_np (np.ndarray): 点云数据
        features_dict (dict): 邻域特征字典
        output_dir (str): 输出目录
        pcd_name (str): 点云名称
        debug (bool): 是否启用调试模式
    """
    if not debug:
        return

    features_dict = np.load(neighbor_features_file, allow_pickle=True).item()
    logging.info("可视化所有邻域特征...")

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
            _visualize_single_feature(
                point_data_np, feature_data, feature_name, output_dir, pcd_name
            )
        else:
            logging.warning(f"邻域特征文件中未找到{feature_name}数据")

    # 创建特征对比可视化
    _visualize_feature_comparison(point_data_np, features_dict, feature_list)


def _visualize_single_feature(
    point_data_np: np.ndarray,
    feature_data: np.ndarray,
    feature_name: str,
    output_dir: str,
    pcd_name: str,
):
    """
    可视化单个特征

    Args:
        point_data_np (np.ndarray): 点云数据
        feature_data (np.ndarray): 特征数据
        feature_name (str): 特征名称
        output_dir (str): 输出目录
        pcd_name (str): 点云名称
    """
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
    output_file = f"{output_dir}/{pcd_name}/{feature_name}_visualization.ply"
    o3d.io.write_point_cloud(output_file, feature_pcd)
    logging.info(f"{feature_name}可视化结果已保存到: {output_file}")

    # 保存特征数据为文本文件
    stats_file = f"{output_dir}/{pcd_name}/{feature_name}_stats.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"{feature_name}特征统计:\n")
        f.write(f"最小值: {np.min(feature_data):.6f}\n")
        f.write(f"最大值: {np.max(feature_data):.6f}\n")
        f.write(f"平均值: {np.mean(feature_data):.6f}\n")
        f.write(f"中位数: {np.median(feature_data):.6f}\n")
        f.write(f"标准差: {np.std(feature_data):.6f}\n")
        f.write(f"数据点数: {len(feature_data)}\n")
    logging.info(f"{feature_name}统计信息已保存到: {stats_file}")


def _visualize_feature_comparison(
    point_data_np: np.ndarray, features_dict: dict, feature_list: list
):
    """
    可视化特征对比

    Args:
        point_data_np (np.ndarray): 点云数据
        features_dict (dict): 特征字典
        feature_list (list): 特征列表
    """
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
            vis.create_window(window_name="邻域特征对比可视化", width=1600, height=900)

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


def visualize_fitting_results(pcd_list: list, debug: bool = False):
    """
    可视化拟合结果

    Args:
        pcd_list (list): 点云列表
        debug (bool): 是否启用调试模式
    """
    if debug and pcd_list:
        o3d.visualization.draw_geometries(pcd_list, window_name="拟合结果可视化")


def visualize_plane_fitting_debug(
    inliers: np.ndarray, outliers: np.ndarray, debug: bool = False
):
    """
    可视化平面拟合调试结果

    Args:
        inliers (np.ndarray): 内点数据
        outliers (np.ndarray): 外点数据
        debug (bool): 是否启用调试模式
    """
    if not debug:
        return

    # 创建内点点云
    debug_pcd_inliers = o3d.geometry.PointCloud()
    debug_pcd_inliers.points = o3d.utility.Vector3dVector(inliers)
    debug_pcd_inliers.paint_uniform_color([255 / 255, 14 / 255, 0 / 255])  # RGB: 红色

    # 创建外点点云
    debug_pcd_outliers = o3d.geometry.PointCloud()
    debug_pcd_outliers.points = o3d.utility.Vector3dVector(outliers)
    debug_pcd_outliers.paint_uniform_color(
        [130 / 255, 178 / 255, 154 / 255]
    )  # RGB: 灰色

    # 可视化
    o3d.visualization.draw_geometries(
        [debug_pcd_inliers, debug_pcd_outliers], window_name="平面拟合调试"
    )
