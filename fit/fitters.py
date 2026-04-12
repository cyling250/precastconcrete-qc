from typing import Optional
import numpy as np
from models import RansacPlaneFitter, RansacLineFitter
from .get_part import get_concrete_part, get_rebar_part
import open3d as o3d
from geometry import BasePlane, calc_inliers_plane
from utils.pcd_utils import calculate_obb_inscribed_cylinder_segment
import logging


def concrete_fit(
    point_data: np.ndarray,
    labels: np.ndarray,
    neighbor_features: str,
    threshold: float = 0.001,
    iterations: int = 1000,
    min_inliers: int = 10,
    max_n_points: int = 4,
    max_plane_num: int = 20,
    random_state: Optional[int] = None,
    concrete_pcd: Optional[np.ndarray] = None,
    debug: bool = False,
):
    """拟合混凝土点云
    Args:
        point_data (np.ndarray): 点云数据
        labels (np.ndarray): 点云聚类标签
        neighbor_features (str): 邻域特征文件路径
        threshold (float, optional): 内点距离阈值. Defaults to None.
        iterations (int, optional): RANSAC最大迭代次数. Defaults to None.
        min_inliers (int, optional): 每次迭代的最小内点数量. Defaults to None.
        max_n_points (int, optional): 每次迭代选取的最大点数量. Defaults to None.
        max_plane_num (int, optional): 混凝土的最大平面数量. Defaults to None.
        random_state (Optional[int], optional): 随机种子. Defaults to None.
    Returns:
        RansacPlaneFitter: RANSAC平面拟合器
    """
    if concrete_pcd is None:
        concrete_pcd = np.asarray(
            get_concrete_part(point_data, labels, neighbor_features).points
        )

    plane_fitter = RansacPlaneFitter(
        threshold=threshold,
        iterations=iterations,
        min_inliers=min_inliers,
        max_n_points=max_n_points,
        max_plane_num=max_plane_num,
        random_state=random_state,
        debug=debug,
    )
    plane_fitter.fit(concrete_pcd)  # 调用RANSAC平面检测函数
    final_plane = []
    final_inliers = []
    # 对混凝土平面进行区分，以免存在多个点云共用一个平面
    concrete_inliers = plane_fitter.get_inliers()
    concrete_plane = plane_fitter.get_plane()
    for i in range(len(concrete_inliers)):
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(concrete_inliers[i])

        # 对点云采用基于距离聚类，将点云分为多个子点云
        labels = pcd_temp.cluster_dbscan(eps=0.05, min_points=10, print_progress=False)
        if len(np.unique(labels)) == 1:
            # 如果只有一个聚类，直接添加到最终结果
            final_inliers.append(concrete_inliers[i])
            final_plane.append(concrete_plane[i])
            continue

        for cluster_id in np.unique(labels):  # 如果点云可以分割为多个子点云，则进行分割
            if cluster_id == -1:  # 如果是噪声点，直接跳过
                continue
            final_inliers.append(concrete_inliers[i][labels == cluster_id])
            # 新创建一个平面，参数为当前平面的参数，主要目的是为了后续区分不同的混凝土平面
            final_plane.append(BasePlane(concrete_plane[i].parameters))

    plane_fitter.set_inliers(final_inliers)
    plane_fitter.set_plane(final_plane)
    return plane_fitter


def rebar_fit(
    point_data: np.ndarray,
    concrete_plane_fitter: RansacPlaneFitter,
    labels: np.ndarray,
    neighbor_features: str,
    neighbot_distance: float = 0.04,
    iterations: int = 5000,
    min_inliers: int = 10,
    max_n_points: int = 10,
    random_state: int = 42,
    concrete_distance_threshold: float = 0.01,
    linearity_threshold: float = 0.6,
) -> list[RansacLineFitter]:
    """拟合钢筋点云
    Args:
        point_data (np.ndarray): 点云数据
        concrete_plane_fitter (RansacPlaneFitter): RANSAC混凝土平面拟合器
        labels (np.ndarray): 点云聚类标签
        neighbor_features (str): 邻域特征文件路径
        threshold (float, optional): 内点距离阈值. Defaults to None.
        iterations (int, optional): RANSAC最大迭代次数. Defaults to None.
        min_inliers (int, optional): 每次迭代的最小内点数量. Defaults to None.
        max_n_points (int, optional): 每次迭代选取的最大点数量. Defaults to None.
        linearity_threshold (float, optional): 线性度阈值. Defaults to None.
        random_state (Optional[int], optional): 随机种子. Defaults to None.
    Returns:
        RansacLineFitter: RANSAC直线拟合器列表
    """
    # 1 从点云中提取钢筋点
    rebars_points = np.asarray(
        get_rebar_part(point_data, labels, neighbor_features).points
    )
    # 2 在钢筋点云上净化非钢筋点
    concrete_inliers = concrete_plane_fitter.get_inliers()
    concrete_plane = concrete_plane_fitter.get_plane()
    for i in range(len(concrete_inliers)):
        # 计算当前平面与所有点的距离, 筛选出距离小于阈值的点
        plane_model = concrete_plane[i]
        inliners_mask = calc_inliers_plane(
            plane_model, rebars_points, concrete_distance_threshold
        )
        # 删除距离大于阈值的点
        rebars_points = rebars_points[~inliners_mask]

    # 3 使用欧式聚类对筛选后的点云进行聚类
    rebars_pcd = o3d.geometry.PointCloud()
    rebars_pcd.points = o3d.utility.Vector3dVector(rebars_points)
    # =============== DEBUG ===============
    # 可视化点云
    # 设置颜色为[233, 122, 94]
    # 获取点云中的点数
    num_points = len(rebars_pcd.points)
    rebars_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([233 / 255, 122 / 255, 94 / 255], (num_points, 1))
    )
    o3d.visualization.draw_geometries([rebars_pcd])
    # =============== END DEBUG ===============
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(rebars_pcd.cluster_dbscan(eps=0.05, min_points=10))

    # 4 对每一类的钢筋进行拟合
    line_fitter_list = []
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        # 筛选当前聚类的点
        cluster_points = rebars_points[labels == cluster_id]

        # 预拟合，目的是为了找到合适的半径
        # 随机采样100次，选择线性度最大的那个来计算半径
        max_linearity = -1
        best_radius = None

        for _ in range(100):
            # 随机选择一点
            random_idx = np.random.randint(0, len(cluster_points))
            random_point = cluster_points[random_idx]
            # 计算距离
            distances = np.linalg.norm(cluster_points - random_point, axis=1)
            neighbor_points = cluster_points[distances < neighbot_distance]

            if len(neighbor_points) < 10:  # 邻域内点数太少，跳过
                continue

            # 计算线性度
            try:
                from utils.eigen_measures_utils import EigenMeasures

                eigen_measures = EigenMeasures(neighbor_points)
                linearity = eigen_measures.linearity

                # 如果线性度更高，更新最佳结果
                if linearity > max_linearity:
                    max_linearity = linearity
                    _, best_radius = calculate_obb_inscribed_cylinder_segment(
                        neighbor_points
                    )
            except Exception as e:
                logging.debug(f"计算线性度失败: {e}")
                continue

        # 如果没有找到合适的半径，使用默认值
        if best_radius is None:
            best_radius = 0.01  # 默认半径
            logging.debug(f"聚类{cluster_id}使用默认半径: {best_radius}")
        else:
            logging.debug(
                f"聚类{cluster_id}的最佳线性度: {max_linearity:.4f}, 邻域半径为{best_radius * 3}"
            )

        # 开始拟合
        line_fitter = RansacLineFitter(
            threshold=best_radius * 3,
            iterations=iterations,
            min_inliers=min_inliers,
            max_n_points=max_n_points,
            random_state=random_state,
            linearity_threshold=linearity_threshold,
            idx=len(line_fitter_list) + 1,
        )
        line_fitter.fit(cluster_points)  # 调用RANSAC平面检测函数
        line_fitter_list.append(line_fitter)
    return line_fitter_list
