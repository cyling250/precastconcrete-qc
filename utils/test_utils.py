import numpy as np
from typing import List, Tuple
from models import RansacPlaneFitter
from geometry import BasePlane


def calculate_residual(
    plane_fitter: RansacPlaneFitter, original_points: np.ndarray
) -> List[float]:
    """
    计算混凝土表面重建的残差指标

    根据研究规范[42, 50]，在真实表面上密集采样点，并评估其与重建表面的对应关系。
    计算每个采样点与其在重建曲面上投影点之间的距离，评估拟合算法对原始曲面几何特征的还原程度。

    参数:
        plane_fitter (RansacPlaneFitter): 混凝土平面拟合器，包含重建的平面信息
        original_points (np.ndarray): 原始混凝土点云数据，形状为(n, 3)

    返回:
        List[float]: 每个重建平面的残差列表，单位为米
    """

    # 获取重建的平面
    reconstructed_planes = plane_fitter.get_plane()

    # 获取每个平面对应的内点
    inliers_list = plane_fitter.get_inliers()

    residuals = []

    for i, plane in enumerate(reconstructed_planes):
        # 获取当前平面对应的内点（作为采样点）
        sample_points = inliers_list[i]

        if len(sample_points) == 0:
            residuals.append(0.0)
            continue

        # 计算每个采样点到重建平面的距离
        distances = []
        for point in sample_points:
            # 计算点到平面的距离
            distance = point_to_plane_distance(point, plane)
            distances.append(distance)

        # 计算平均距离作为该平面的残差
        avg_distance = np.mean(distances)
        residuals.append(avg_distance)

    return residuals


def calculate_p_coverage(
    plane_fitter: RansacPlaneFitter,
    original_points: np.ndarray,
    distance_threshold: float = 0.01,
) -> float:
    """
    计算混凝土表面重建的P-覆盖率指标

    量化生成表面所覆盖输入点云的比例。

    参数:
        plane_fitter (RansacPlaneFitter): 混凝土平面拟合器，包含重建的平面信息
        original_points (np.ndarray): 原始混凝土点云数据，形状为(n, 3)
        distance_threshold (float): 距离阈值，依据先前研究设定为0.01米

    返回:
        float: P-覆盖率，范围[0, 1]
    """

    # 获取重建的平面
    reconstructed_planes = plane_fitter.get_plane()

    # 获取所有内点（即被重建表面覆盖的点）
    all_inliers = []
    inliers_list = plane_fitter.get_inliers()
    for inliers in inliers_list:
        all_inliers.extend(inliers)

    if len(all_inliers) == 0:
        return 0.0

    # 将内点列表转换为numpy数组
    all_inliers_array = np.array(all_inliers)

    # 计算被覆盖的点数
    covered_count = 0

    for point in original_points:
        # 计算点到所有重建平面的最小距离
        min_distance = float("inf")

        for plane in reconstructed_planes:
            distance = point_to_plane_distance(point, plane)
            if distance < min_distance:
                min_distance = distance

        # 如果最小距离小于阈值，则认为该点被覆盖
        if min_distance <= distance_threshold:
            covered_count += 1

    # 计算覆盖率
    p_coverage = covered_count / len(original_points)

    return p_coverage


def point_to_plane_distance(point: np.ndarray, plane: BasePlane) -> float:
    """
    计算点到平面的距离

    参数:
        point (np.ndarray): 点坐标，形状为(3,)
        plane (BasePlane): 平面对象

    返回:
        float: 点到平面的距离，单位为米
    """

    if plane.parameters is None:
        raise ValueError("平面参数未初始化")

    # 提取平面参数
    c = plane.parameters[0:3]  # 平面上的点
    n = plane.parameters[3:6]  # 平面法向量（已归一化）

    # 计算点到平面的距离公式: |(p - c) · n|
    distance = abs(np.dot(point - c, n))

    return distance
