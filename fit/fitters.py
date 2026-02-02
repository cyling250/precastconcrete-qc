from typing import Optional
import numpy as np
from models import RansacPlaneFitter, RansacLineFitter, calc_inliers_plane, BasePlane
from .get_part import get_concrete_part, get_rebar_part
import open3d as o3d


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
        labels = pcd_temp.cluster_dbscan(eps=0.05, min_points=10, print_progress=debug)
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
            final_plane.append(BasePlane(concrete_plane[i].get_parameters()))

        if debug:
            # DEBUG 可视化当前点云
            o3d.visualization.draw_geometries(final_inliers)
    plane_fitter.set_inliers(final_inliers)
    plane_fitter.set_plane(final_plane)
    return plane_fitter


def rebar_fit(
    point_data: np.ndarray,
    concrete_plane_fitter: RansacPlaneFitter,
    labels: np.ndarray,
    neighbor_features: str,
    threshold: float = 0.006,
    iterations: int = 5000,
    min_inliers: int = 10,
    max_n_points: int = 10,
    random_state: int = 42,
    linearity_threshold: float = 0.6,
    debug: bool = False,
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
        inliners_mask = calc_inliers_plane(plane_model, rebars_points, 0.01)
        # 删除距离大于阈值的点
        rebars_points = rebars_points[~inliners_mask]

    # DEBUG 可视化
    if debug:
        rebars_pcd = o3d.geometry.PointCloud()
        rebars_pcd.points = o3d.utility.Vector3dVector(rebars_points)
        o3d.visualization.draw_geometries([rebars_pcd])

    # 3 使用欧式聚类对筛选后的点云进行聚类
    rebars_pcd = o3d.geometry.PointCloud()
    rebars_pcd.points = o3d.utility.Vector3dVector(rebars_points)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(rebars_pcd.cluster_dbscan(eps=0.05, min_points=10))

    # 4 对每一类的钢筋进行拟合
    line_fitter_list = []
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        # 筛选当前聚类的点
        cluster_points = rebars_points[labels == cluster_id]
        line_fitter = RansacLineFitter(
            threshold=threshold,
            iterations=iterations,
            min_inliers=min_inliers,
            max_n_points=max_n_points,
            random_state=random_state,
            linearity_threshold=linearity_threshold,
        )
        line_fitter.fit(cluster_points)  # 调用RANSAC平面检测函数
        line_fitter_list.append(line_fitter)
    return line_fitter_list


def visualize_plane_fitter(plane_fitter):
    # 可视化
    inliers_pcd_list = []
    for inlier in plane_fitter.get_inliers():
        inliers_pcd = o3d.geometry.PointCloud()
        inliers_pcd.points = o3d.utility.Vector3dVector(inlier)
        # 为每个点生成随机颜色
        random_color = np.random.rand(3)  # 生成随机RGB颜色
        inliers_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(random_color, (len(inlier), 1))
        )
        inliers_pcd_list.append(inliers_pcd)
    o3d.visualization.draw_geometries(inliers_pcd_list)
    return


def visualize_line_fitter(line_fitter):
    """可视化RANSAC直线拟合结果
    Args:
        plane_fitter (RansacLineFitter): RANSAC直线拟合器
    """
    inliers = line_fitter.get_inliers()
    outliers = line_fitter.get_outliers()

    # 可视化当前聚类的点云
    linear_equation = line_fitter.get_line_equation()
    inliers_pcd_list = []
    line_list = []

    for i, (inlier, line_eq) in enumerate(zip(inliers, linear_equation)):
        # 创建内点点云
        inliers_pcd = o3d.geometry.PointCloud()
        inliers_pcd.points = o3d.utility.Vector3dVector(inlier)
        # 为每个平面生成随机颜色
        random_color = np.random.rand(3)  # 生成随机RGB颜色
        inliers_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(random_color, (len(inlier), 1))
        )
        inliers_pcd_list.append(inliers_pcd)

        # 创建直线几何体
        cx, cy, cz, nx, ny, nz = line_eq

        # 计算直线的起点和终点（基于内点的范围）
        if len(inlier) > 0:
            # 计算内点在直线方向上的投影坐标
            inlier_centered = inlier - np.array([cx, cy, cz])
            projections = np.dot(inlier_centered, np.array([nx, ny, nz]))
            min_proj = np.min(projections)
            max_proj = np.max(projections)

            # 扩展一定范围，使直线更长
            extension = 0.1 * (max_proj - min_proj)
            min_proj -= extension
            max_proj += extension

            # 计算起点和终点
            start_point = np.array([cx, cy, cz]) + min_proj * np.array([nx, ny, nz])
            end_point = np.array([cx, cy, cz]) + max_proj * np.array([nx, ny, nz])

            # 创建直线
            line_points = np.array([start_point, end_point])
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(line_points)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([random_color])
            line_list.append(line)

    outliers_pcd = o3d.geometry.PointCloud()
    outliers_pcd.points = o3d.utility.Vector3dVector(outliers)
    outliers_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0, 0, 1], (len(outliers), 1))
    )

    # 可视化点云和直线
    o3d.visualization.draw_geometries(inliers_pcd_list + [outliers_pcd] + line_list)


def close_rebar(line_fitter):
    """给钢筋封口
    Args:
        line_fitter (RansacLineFitter): RANSAC直线拟合器
    Returns:
        list[np.ndarray]: 封口点云列表
    """
    all_point = []
    line_equation_list = line_fitter.get_line_equation()
    inliers_list = line_fitter.get_inliers()
    for i in range(len(inliers_list)):
        current_inliers = inliers_list[i]
        if len(current_inliers) == 0:
            continue

        line_eq = line_equation_list[i]
        cx, cy, cz, nx, ny, nz = line_eq
        direction = np.array([nx, ny, nz])

        # 计算内点在直线方向上的投影坐标
        inlier_centered = current_inliers - np.array([cx, cy, cz])
        projections = np.dot(inlier_centered, direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)

        # 找到两个最远点（封头位置）
        min_point = np.array([cx, cy, cz]) + min_proj * direction
        max_point = np.array([cx, cy, cz]) + max_proj * direction

        # 构造两个封头平面（法向量为直线方向向量）
        # 平面1：通过min_point，法向量为direction
        # 平面2：通过max_point，法向量为direction

        # 将内点向两个平面投影
        # 投影到min_point平面
        projections_min = np.dot(current_inliers - min_point, direction)
        cap_points_min = current_inliers - projections_min[:, np.newaxis] * direction

        # 投影到max_point平面
        projections_max = np.dot(current_inliers - max_point, direction)
        cap_points_max = current_inliers - projections_max[:, np.newaxis] * direction

        # 将封头点云添加到结果中
        all_point.append(cap_points_min)
        all_point.append(cap_points_max)
    return all_point
