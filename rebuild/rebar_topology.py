from models.plane_fitter import RansacPlaneFitter
import numpy as np
from geometry import (
    SegmentLine,
    intersect_line,
    project_points_to_line,
    BaseLine,
    BasePoint,
    distance_point_to_line,
    project_points_to_line_segment,
)
from models import RansacLineFitter
import logging
from utils.eigen_measures_utils import EigenMeasures
from utils.pcd_utils import calculate_obb_inscribed_cylinder_segment


def get_rebar_param_by_line_fitter(
    line_fitters: list[RansacLineFitter],
    max_distance: float = 0.03,
) -> list[list[SegmentLine, float]]:
    """
    根据 RansacLineFitter 获取钢筋参数

    参数:
    line_fitters: list[RansacLineFitter], RansacLineFitter 列表
    max_distance: float, 最大距离，默认 0.03

    返回:
    rebar_param_list: list[list[SegmentLine, float]], 钢筋参数列表，每个元素为 [SegmentLine, 直径]
    """
    # 准备数据
    rebar_param_list = []
    # 构造线段模型
    for line_fitter in line_fitters:
        # 准备数据
        inliers_point_list: list[np.ndarray] = line_fitter.get_inliers()
        line_list: list[BaseLine] = line_fitter.get_lines()

        segment_line_list: list[SegmentLine] = []
        for line in line_list:
            segment_line_list.append(SegmentLine(line.parameters))

        for i in range(len(line_list)):
            # 如果点云的线性度小于0.5，跳过
            linearity = EigenMeasures(inliers_point_list[i]).linearity
            logging.info(f"点云{i}的线性度为{linearity}")
            if linearity < 0.5:
                continue
            for j in range(len(line_list)):
                if i == j:
                    # 跳过自己
                    continue
                # 计算直线的交点
                cross_point = intersect_line(line_list[i], line_list[j])
                if cross_point is None:
                    # 直线不相交，跳过
                    continue
                # 计算点云到交点的距离
                distances = np.linalg.norm(
                    inliers_point_list[i] - cross_point.location, axis=1
                )
                min_distance = np.min(distances)
                if min_distance > max_distance:
                    # 点云到交点的距离大于最大距离，跳过
                    continue
                # 构造SegmentLine
                if cross_point not in segment_line_list[i].link_points:
                    try:
                        segment_line_list[i].add_link_point(cross_point)
                    except ValueError:
                        continue

        # 构建SegmentLine
        for i in range(len(segment_line_list)):
            segment_line = segment_line_list[i]
            if None in segment_line.link_points:
                # 先将点云向直线投影
                line_pcd_proj = project_points_to_line(
                    inliers_point_list[i],
                    segment_line.parameters[:3],
                    segment_line.parameters[3:],
                )
                # 计算t值
                t_list = segment_line.get_point_lambda(line_pcd_proj)
                t_min = np.min(t_list) - 0.01
                t_max = np.max(t_list) + 0.01
                # 得到两个端点
                p1 = segment_line.parameters[:3] + t_min * segment_line.parameters[3:]
                p2 = segment_line.parameters[:3] + t_max * segment_line.parameters[3:]
                # 构造SegmentLine
                segment_line.link_points = [None, None]
                segment_line.add_link_point(BasePoint(p1))
                segment_line.add_link_point(BasePoint(p2))
            else:
                continue
            # 计算点云到直线的距离
            distances = distance_point_to_line(
                inliers_point_list[i],
                segment_line.parameters[:3],
                segment_line.parameters[3:],
            )
            # 取中位数作为半径
            radius = np.median(distances)
            rebar_param_list.append([segment_line, radius])
    return rebar_param_list


def get_rebar_param_by_line_fitter_v2(
    line_fitters: list[RansacLineFitter],
    plane_fitter: list[RansacPlaneFitter],
    max_distance: float = 0.07,
    **kwargs,
) -> list[list[SegmentLine, float]]:
    """
    根据 RansacLineFitter 获取钢筋参数，使用OBB包围盒计算内接圆柱体

    参数:
    line_fitters: list[RansacLineFitter], RansacLineFitter 列表
    max_distance: float, 最大距离阈值，默认0.07
    **kwargs: 其他参数，传递给 get_rebar_param_by_line_fitter

    返回:
    rebar_param_list: list[list[SegmentLine, float]], 钢筋参数列表，每个元素为 [SegmentLine, 半径]
    """
    # 准备数据
    rebar_param_list = []
    count = 0
    # 收集所有直线和对应的点云
    for line_fitter in line_fitters:
        count += 1
        rebar_params = []
        inliers_point_list = line_fitter.get_inliers()
        line_list = line_fitter.get_lines()
        # 计算交点并更新点云
        for i in range(len(line_list)):
            for j in range(i + 1, len(line_list)):
                # 计算直线交点
                cross_point = intersect_line(line_list[i], line_list[j])
                if cross_point is None:
                    continue

                # 计算交点距离两条直线对应点云的最小距离
                distances_i = np.linalg.norm(
                    inliers_point_list[i] - cross_point.location, axis=1
                )
                distances_j = np.linalg.norm(
                    inliers_point_list[j] - cross_point.location, axis=1
                )
                min_distance_i = np.min(distances_i)
                min_distance_j = np.min(distances_j)
                # 如果交点距离两条直线的最小距离都小于阈值，将交点添加到点云中
                if min_distance_i < max_distance and min_distance_j < max_distance:
                    # 将交点添加到对应的点云中
                    inliers_point_list[i] = np.vstack(
                        [inliers_point_list[i], cross_point.location]
                    )
                    inliers_point_list[j] = np.vstack(
                        [inliers_point_list[j], cross_point.location]
                    )
                    logging.debug(f"count{count}:在直线{i}和{j}之间找到交点")

        # 计算每个更新后的点云的OBB内接圆柱体参数
        for i in range(len(line_list)):
            points = inliers_point_list[i]
            line: BaseLine = line_list[i]
            line_direction = line.parameters[:3]

            # 使用统一工具函数计算OBB包围盒内接圆柱体
            try:
                segment_line, radius = calculate_obb_inscribed_cylinder_segment(
                    points, line_direction
                )
                # 将点云投影到线上，并得到距离最远的两个点
                real_line_params = list(segment_line.link_points[0].location) + list(
                    segment_line.link_points[1].location
                    - segment_line.link_points[0].location
                )
                p1, p2 = project_points_to_line_segment(
                    points, BaseLine(np.array(real_line_params))
                )
                segment_line = SegmentLine(np.array(real_line_params), [p1, p2])
                rebar_params.append([segment_line, radius * 1.2])
            except ValueError as e:
                logging.error(f"计算钢筋参数失败: {e}")
                continue
        rebar_param_list.append(rebar_params)

    return rebar_param_list
