import numpy as np
from geometry import (
    BasePlane,
    SegmentLine,
    CrossLine,
    intersect_plane_with_pcd,
    intersect_line,
    BasePoint,
)
import open3d as o3d
import logging


def get_cross_lines_by_plane(
    pcds: list[o3d.geometry.PointCloud], planes: list[BasePlane]
) -> list[CrossLine]:
    """计算所有平面之间的交线

    通过遍历所有平面组合，计算每对平面之间的交线，并基于点云数据确定交线的有效范围。

    Args:
        pcds (list[o3d.geometry.PointCloud]): 点云列表，每个点云对应一个平面
        planes (list[BasePlane]): 平面模型列表，与点云列表一一对应

    Returns:
        list[CrossLine]: 所有有效的交线列表，包含交线的参数和起终点信息
    """
    # 数据校验
    if len(pcds) != len(planes):
        raise ValueError("点云列表和平面列表长度必须一致")

    cross_lines: list[CrossLine] = []
    for i in range(len(pcds)):
        for j in range(i + 1, len(pcds)):
            # 计算平面的交线
            cross_line: CrossLine = intersect_plane_with_pcd(
                planes[i], planes[j], pcds[i].points, pcds[j].points
            )
            if cross_line is not None:
                cross_lines.append(cross_line)
    return cross_lines


def visualize_cross_lines(cross_lines: list[CrossLine], debug: bool = False):
    """可视化交线

    Args:
        cross_lines (list[CrossLine]): 交线列表
    """
    if not debug:
        return
    geometries = []
    for line in cross_lines:
        line_pcd = o3d.geometry.PointCloud()
        if isinstance(line, CrossLine):
            line_pcd_point = np.linspace(line.point_start, line.point_end, num=1000)
        elif isinstance(line, SegmentLine):
            if line.link_points[0] is None or line.link_points[1] is None:
                logging.warning(f"线段{line}的起(终)点为空")
                continue
            line_pcd_point = np.linspace(
                line.link_points[0].location, line.link_points[1].location, num=1000
            )
        line_pcd.points = o3d.utility.Vector3dVector(line_pcd_point)
        # 交线赋值为红色
        line_pcd.paint_uniform_color([1, 0, 0])
        geometries.append(line_pcd)
    o3d.visualization.draw_geometries(geometries)


def get_cross_points_by_cross_lines(
    cross_lines: list[CrossLine],
    expand_list: list[float] = [0, 0.01, 0.02, 0.03, 0.04, 0.05],
) -> tuple[list[BasePoint], list[SegmentLine]]:
    """根据交线计算交点

    Args:
        cross_lines (list[CrossLine]): 交线列表
        expand_list (list[float]): 扩展参数列表，用于计算交线的交点，默认值为[0, 0.01, 0.02, 0.03, 0.04, 0.05]

    Returns:
        tuple[list[BasePoint], list[SegmentLine]]: 所有交线的交点列表和交线列表
    """
    # 初始化线段列表，长度和交线列表一致
    segment_lines: list[SegmentLine] = [
        SegmentLine(line.parameters) for line in cross_lines
    ]
    for i in range(len(cross_lines)):
        # 按照0.01的扩展参数计算交线的交点
        real_link_point_complete = False
        for expand in expand_list:
            if real_link_point_complete:
                break
            for j in range(len(cross_lines)):
                if None not in segment_lines[i].link_points:
                    # 当得到2个交点时，跳出循环
                    real_link_point_complete = True
                    break
                if i == j:
                    # 交线i与交线j是同一个交线，跳过
                    continue
                # 计算交线的交点
                cross_point: BasePoint | None = intersect_line(
                    cross_lines[i], cross_lines[j], expand=expand
                )
                # 保存交点
                if cross_point is not None:
                    logging.debug(f"交线{i}与交线{j}的交点为{cross_point}")
                    segment_lines[i].add_link_point(cross_point)

    points: list[BasePoint] = []
    for line in segment_lines:
        for point in line.link_points:
            if point not in points:
                points.append(point)

    # 数据校验，要求每个交线的连接点数量为2
    for line in segment_lines:
        if None in line.link_points:
            raise ValueError(f"交线{line}的连接点数量不为2")
    return points, segment_lines
