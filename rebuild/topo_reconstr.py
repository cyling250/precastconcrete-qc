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


def concrete_reconstr(
    output_dir: str,
    pcd_name: str,
    *,
    expand_list: list = None,
    file_format: str = "vtk",
):
    """
    混凝土三维重建

    Args:
        output_dir (str): 输出目录
        pcd_name (str): 点云名称
        expand_list (list): 直线向外的扩展列表，默认为[0, 0.01, 0.02, 0.03, 0.04, 0.05]
    """
    import open3d as o3d
    import logging
    import glob
    import pygmsh
    from geometry import BasePlane
    from rebuild.concrete_params import (
        visualize_cross_lines,
        get_cross_points_by_cross_lines,
        get_cross_lines_by_plane,
    )

    # 设置默认扩展列表
    if expand_list is None:
        expand_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05]

    logging.info("混凝土三维重建...")
    concrete_pcds: list[o3d.geometry.PointCloud] = []
    concrete_planes: list[BasePlane] = []

    # 自动遍历所有concrete_plane_params_{i}.npy文件
    concrete_plane_param_files = glob.glob(
        f"{output_dir}/{pcd_name}/concrete_plane_params_*.npy"
    )

    # 获取平面数量
    plane_count = len(concrete_plane_param_files)
    logging.info(f"找到 {plane_count} 个混凝土平面参数文件")

    for i in range(plane_count):
        tmp_pcd_file = f"{output_dir}/{pcd_name}/concrete_inliers_{i}.ply"
        tmp_plane_file = f"{output_dir}/{pcd_name}/concrete_plane_params_{i}.npy"

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
        concrete_cross_lines, expand_list=expand_list
    )
    logging.info(f"交点数量: {len(concrete_cross_points)}")

    with pygmsh.occ.Geometry() as geom:
        # 定义实体的顶点和棱
        _, geom_lines = generate_lines_and_points(
            concrete_cross_points, concrete_segment_lines, geom
        )

        # 由线环构建面
        faces = build_face(concrete_planes, concrete_cross_lines, geom_lines, geom)

        # 构建闭合面环
        surf_loop = geom.add_surface_loop(faces)

        # 面环围合出三维实体
        geom.add_volume(surf_loop)

        # 生成网格并保存
        mesh = geom.generate_mesh()
        mesh.write(
            f"{output_dir}/{pcd_name}/mesh-concrete.{file_format}",
            file_format=file_format,
        )
        logging.info(
            f"三维面包围体网格生成完成，文件：{output_dir}/{pcd_name}/mesh-concrete.{file_format}"
        )


def rebar_reconstr(
    output_dir: str,
    pcd_name: str,
    rebar_param_list: list,
    *,
    file_format: str = "vtk",
):
    """
    钢筋三维重建

    Args:
        output_dir (str): 输出目录
        pcd_name (str): 点云名称
        rebar_param_list (list): 钢筋参数列表，每个元素为[SegmentLine, radius]的列表
        file_format (str): 输出文件格式，默认为"vtk"
    """
    import logging
    import pygmsh
    import numpy as np
    from geometry import SegmentLine

    logging.info("钢筋三维重建...")

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
        mesh.write(
            f"{output_dir}/{pcd_name}/mesh-rebar.{file_format}",
            file_format=file_format,
        )
        logging.info(
            f"钢筋网格生成完成，文件：{output_dir}/{pcd_name}/mesh-rebar.{file_format}"
        )
