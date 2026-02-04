import numpy as np
import open3d as o3d
import pygmsh
import pyvista as pv
from typing import List
from geometry import (
    CrossLine,
    CrossPoint,
    BasePlane,
    intersect_line,
    intersect_plane_with_pcd,
)

# =====================读取文件=====================
# test_name = "prefabricated_laminated_beam"
test_name = "prefabricated_column"
pcds: list[o3d.geometry.PointCloud] = []
planes = []
for i in range(6):
    # 读取点云
    pcd = o3d.io.read_point_cloud(f"output/{test_name}/concrete_inliers_{i}.ply")
    pcds.append(pcd)

    # 读取平面参数方程
    plane: BasePlane = BasePlane.read_from_npy(
        f"output/{test_name}/concrete_plane_params_{i}.npy"
    )
    planes.append(plane)

# =====================计算所有平面的真实交线=====================
cross_lines: list[CrossLine] = []
for i in range(len(pcds)):
    for j in range(i + 1, len(pcds)):
        # 计算平面的交线
        cross_line: CrossLine = intersect_plane_with_pcd(
            planes[i], planes[j], pcds[i].points, pcds[j].points
        )
        if cross_line is not None:
            print(f"点云{i}与点云{j}的交线为{cross_line.parameters}")
            cross_lines.append(cross_line)
# =====================可视化所有交线=====================
geometries = []
for line in cross_lines:
    line_pcd = o3d.geometry.PointCloud()
    line_pcd_point = np.linspace(line.point_start, line.point_end, num=1000)
    line_pcd.points = o3d.utility.Vector3dVector(line_pcd_point)
    # 交线赋值为红色
    line_pcd.paint_uniform_color([1, 0, 0])
    geometries.append(line_pcd)
# 可视化
if geometries:
    o3d.visualization.draw_geometries(geometries, window_name="所有交线的可视化")

# =====================计算所有交线的交点=====================
point_rebuild_pcd = []
for i in range(len(cross_lines)):
    # 按照0.01的扩展参数计算交线的交点
    real_link_point_complete = False
    for expand in [0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        if real_link_point_complete:
            break
        for j in range(len(cross_lines)):
            if len(cross_lines[i].link_points) == 2:
                # 当得到2个交点时，跳出循环
                real_link_point_complete = True
                break
            if i == j:
                # 交线i与交线j是同一个交线，跳过
                continue
            # 计算交线的交点
            cross_point: CrossPoint | None = intersect_line(
                cross_lines[i], cross_lines[j], expand=expand
            )
            # 保存交点
            if cross_point is not None:
                print(f"交线{i}与交线{j}的交点为{cross_point.location}")
                if cross_point not in cross_lines[i].link_points:
                    cross_lines[i].link_points.append(cross_point)

cross_points: List[CrossPoint] = []
for line in cross_lines:
    for point in line.link_points:
        if point not in cross_points:
            cross_points.append(point)

print(f"共读取{len(planes)}个面")
print(f"共计算出{len(cross_lines)}条交线")
print(f"共得到{len(cross_points)}个交点")

# =====================可视化所有交点和连接的交线=====================
geometries = []
for line in cross_lines:
    line_pcd = o3d.geometry.PointCloud()
    line_pcd_point = np.linspace(
        line.link_points[0].location, line.link_points[1].location, num=1000
    )
    line_pcd.points = o3d.utility.Vector3dVector(line_pcd_point)
    # 交线赋值为红色
    line_pcd.paint_uniform_color([1, 0, 0])
    geometries.append(line_pcd)
o3d.visualization.draw_geometries(geometries)

# 数据校验，要求每个交线的连接点数量为2
for line in cross_lines:
    if len(line.link_points) != 2:
        raise ValueError(f"交线{line}的连接点数量不为2，实际为{len(line.link_points)}")
with pygmsh.geo.Geometry() as geom:
    # 第一步：定义实体的顶点
    geom_points = []  # 存储所有有效点的几何点对象 和 cross_points中的点有序对应
    for point in cross_points:
        geom_points.append(geom.add_point(point.location, mesh_size=0.2))

    # 第二步，定义实体的棱
    geom_lines = []  # 存储所有有效线的几何线对象 和 cross_lines中的线有序对应
    for line in cross_lines:
        link_points: list[int] = []
        point1_idx = np.where(np.array(cross_points) == line.link_points[0])[0][0]
        point2_idx = np.where(np.array(cross_points) == line.link_points[1])[0][0]
        link_points.append(point1_idx)
        link_points.append(point2_idx)
        geom_lines.append(
            geom.add_line(geom_points[link_points[0]], geom_points[link_points[1]])
        )

    # 第三步：由线环构建面
    loops = []  # 存储所有有效面环的几何面环对象 和 planes中的面有序对应
    faces = []  # 存储所有有效面的几何面对象

    for plane in planes:
        link_lines: list[int] = []
        # 遍历交线列表，找出组成该面的交线
        for i in range(len(cross_lines)):
            if plane == cross_lines[i].plane1 or plane == cross_lines[i].plane2:
                link_lines.append(i)

        # 收集属于该面的所有几何线
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
        def calculate_loop_area(loop_lines):
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
                    except:
                        return 0.0
            # 确保闭合
            if points and points[0] != points[-1]:
                points.append(points[0])

            # 计算面积
            area = 0.0
            n = len(points)
            if n >= 3:
                for i in range(n - 1):
                    x1, y1, z1 = points[i]
                    x2, y2, z2 = points[i + 1]
                    area += x1 * y2 - x2 * y1
                return abs(area) / 2.0
            else:
                # 如果点不够，返回线的数量作为面积的近似
                return len(loop_lines)

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
        loops.append(main_loop)
        # 保存孔洞环
        for hole_loop in hole_loops:
            loops.append(hole_loop)

    # 第四步：构建闭合面环（核心：面包围体的关键，将6个面组合成封闭面环）
    surf_loop = geom.add_surface_loop(faces)

    # 第五步：面环围合出三维实体（核心操作：add_volume入参为闭合面环）
    volume_3d = geom.add_volume(surf_loop)  # 面包围得到的三维体实体

    # 第六步：生成网格并保存
    mesh = geom.generate_mesh()
    mesh.write(f"output/{test_name}/mesh.vtk", file_format="vtk")

print(f"三维面包围体网格生成完成，文件：output/{test_name}/mesh.vtk")


# 直接读取之前导出的网格可视化
grid = pv.read(f"output/{test_name}/mesh.vtk")
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="white")
plotter.add_axes(line_width=2)
# plotter.camera_position = [(2.0, 2.0, 1.0), (0.5, 0.3, 0.0), (0.0, 0.0, 1.0)]
plotter.show()
