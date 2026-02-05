from geometry import BasePoint, CrossLine, BasePlane, SegmentLine
import pygmsh
import numpy as np


def calculate_loop_area(loop_lines):
    """
    计算环的面积（基于多边形面积公式）

    Args:
        loop_lines (list[pygmsh.geo.Line]): 环中的线列表

    Returns:
        float: 环的面积
    """

    # 简单计算环的面积（基于多边形面积公式）
    points = []
    for line in loop_lines:
        # 获取点的坐标，pygmsh中的Point对象通常有x、y、z属性
        point = line.points[0]
        if hasattr(point, "x") and hasattr(point, "y"):
            # 对于2D点
            points.append([point.x, point.y, 0.0])
        elif hasattr(point, "coordinates"):
            # 备用方案
            points.append(point.coordinates)
        else:
            # 尝试直接获取坐标（根据pygmsh版本不同可能有不同实现）
            try:
                # 对于pygmsh的Point对象，可能需要通过其他方式获取坐标
                # 这里使用一个简单的替代方案，基于线的长度来估算面积
                return len(loop_lines)  # 简化处理，使用线的数量作为面积的近似
            except Exception as e:
                print(f"Error calculating loop area: {e}")
                return 0.0
    # 确保闭合
    if points and points[0] != points[-1]:
        points.append(points[0])

    # 计算面积
    area = 0.0
    n = len(points)
    if n >= 3:
        for i in range(n - 1):
            x1, y1, _ = points[i]
            x2, y2, _ = points[i + 1]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
    else:
        # 如果点不够，返回线的数量作为面积的近似
        return len(loop_lines)


def generate_lines_and_points(
    points: list[BasePoint],
    segment_lines: list[SegmentLine],
    geom: pygmsh.geo.Geometry,
):
    """
    生成交线的点和线的几何对象

    Args:
        cross_points (list[BasePoint]): 交线的交点列表
        segment_lines (list[SegmentLine]): 交线的列表
    """
    # 定义实体的顶点和棱
    geom_points = []  # 存储所有有效点的几何点对象 和 points中的点有序对应
    for point in points:
        geom_points.append(geom.add_point(point.location, mesh_size=0.2))

    # 定义实体的棱
    geom_lines = []  # 存储所有有效线的几何线对象 和 segment_lines中的线有序对应
    for line in segment_lines:
        link_points: list[int] = []
        point1_idx = np.where(np.array(points) == line.link_points[0])[0][0]
        point2_idx = np.where(np.array(points) == line.link_points[1])[0][0]
        link_points.append(point1_idx)
        link_points.append(point2_idx)
        line_geom = geom.add_line(
            geom_points[link_points[0]], geom_points[link_points[1]]
        )
        geom_lines.append(line_geom)
    return geom_points, geom_lines


def build_face(
    planes: list[BasePlane],
    cross_lines: list[CrossLine],
    geom_lines: list,
    geom: pygmsh.geo.Geometry,
):
    """
    由交线构建面

    Args:
        planes (list[BasePlane]): 平面列表
        cross_lines (list[CrossLine]): 交线的列表
        segment_lines (list[SegmentLine]): 线段的列表 和 cross_lines中的线有序对应
        geom_lines (list[pygmsh.geo.Line]): 几何线的列表
    """
    faces = []
    for plane in planes:
        link_lines: list[int] = []
        # 遍历交线列表，找出组成该面的交线
        for i in range(len(cross_lines)):
            if plane == cross_lines[i].plane1 or plane == cross_lines[i].plane2:
                link_lines.append(i)
        plane_geom_lines = [geom_lines[i] for i in link_lines]
        remaining_lines = plane_geom_lines.copy()

        # 识别所有环
        all_loops = []
        while remaining_lines:
            # 开始构建一个新环
            current_loop = []
            current_line = remaining_lines.pop(0)
            current_loop.append(current_line)

            # 构建闭合环
            while True:
                current_end = current_line.points[1]
                found_next = False

                for i, line in enumerate(remaining_lines):
                    if line.points[0] == current_end:
                        # 线的起点与当前环的终点匹配
                        current_line = line
                        current_loop.append(current_line)
                        remaining_lines.pop(i)
                        found_next = True
                        break
                    elif line.points[1] == current_end:
                        # 线的终点与当前环的终点匹配，需要反转线
                        current_line = -line
                        current_loop.append(current_line)
                        remaining_lines.pop(i)
                        found_next = True
                        break

                if not found_next:
                    # 检查环是否闭合
                    if current_loop[0].points[0] == current_loop[-1].points[1]:
                        all_loops.append(current_loop)
                    break

        # 计算每个环的面积
        loop_areas = [(calculate_loop_area(loop), loop) for loop in all_loops]

        # 按面积排序，最大的作为主环，其余作为孔洞
        loop_areas.sort(key=lambda x: x[0], reverse=True)

        if not loop_areas:
            raise ValueError(f"面{plane}没有找到有效环")

        # 主环
        main_loop = geom.add_curve_loop(loop_areas[0][1])

        # 孔洞环
        hole_loops = []
        for _, loop in loop_areas[1:]:
            hole_loop = geom.add_curve_loop(loop)
            hole_loops.append(hole_loop)

        # 构建面
        face = geom.add_plane_surface(main_loop, hole_loops)
        faces.append(face)
    return faces
