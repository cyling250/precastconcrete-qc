from models.line_model import CrossLine
from models.line_model import CrossPoint
import open3d as o3d
import numpy as np
from models.plane_model import BasePlane, intersect_plane
from models.line_model import intersect_line

# 读取文件
test_name = "prefabricated_laminated_beam"
# test_name = "prefabricated_column"
pcds = []
planes = []
for i in range(25):
    # 读取点云
    pcd = o3d.io.read_point_cloud(f"output/{test_name}/concrete_inliers_{i}.ply")
    pcds.append(pcd)

    # 读取平面参数方程
    plane: BasePlane = BasePlane.read_from_npy(
        f"output/{test_name}/concrete_plane_params_{i}.npy"
    )
    planes.append(plane)

# 计算所有平面的真实交线
cross_lines: list[CrossLine] = []
for i in range(len(pcds)):
    for j in range(i + 1, len(pcds)):
        plane = planes[i]
        # 计算平面的交线
        cross_line = intersect_plane(plane, planes[j])
        if cross_line is None:
            print(f"平面{i}与平面{j}平行，无交线")
        else:
            if cross_line.in_cloud(pcds[i].points):
                cross_lines.append(cross_line)
                print(f"平面{i}与平面{j}的交线在点云{i}中")
            else:
                print(f"平面{i}与平面{j}的交线不在点云{i}中")

# 计算所有交线的交点
cross_points: list[CrossPoint] = []
for i in range(len(cross_lines)):
    for j in range(i + 1, len(cross_lines)):
        cross_point = intersect_line(cross_lines[i], cross_lines[j])
        if cross_point is None:
            # print(f"交线{i}与交线{j}平行，无交点")
            pass
        else:
            point_exists = False
            for point in cross_points:
                if cross_point.equal(point):
                    point.link_lines.add(cross_lines[i])
                    point.link_lines.add(cross_lines[j])
                    point_exists = True
                    break
            if not point_exists:
                cross_points.append(cross_point)
            print(f"交线{i}与交线{j}的交点为{cross_point.get_location()}")

# 判断点距离点云的距离小于阈值
threshold = 0.1  # 取10cm即可，因为误差而被误判平行的点交点通常与点云距离超过10cm
valid_points: list[CrossPoint] = []
for point in cross_points:
    for pcd in pcds:
        distances = np.linalg.norm(pcd.points - point.get_location(), axis=1)
        if np.min(distances) < threshold:
            valid_points.append(point)
            break

print(f"共读取{len(planes)}个面")
print(f"共计算出{len(cross_lines)}条交线")
print(f"去重后共得到{len(valid_points)}个交点")
# 重建三维模型
import pygmsh

with pygmsh.geo.Geometry() as geom:
    # 第一步：定义实体的顶点
    geom_points = []  # 存储所有有效点的几何点对象 和 valid_points中的点有序对应
    for point in valid_points:
        geom_points.append(geom.add_point(point.get_location(), mesh_size=0.2))

    # 第二步，定义实体的棱
    geom_lines = []  # 存储所有有效线的几何线对象 和 cross_lines中的线有序对应
    for line in cross_lines:
        link_points: list[int] = []
        for i in range(len(valid_points)):
            # 遍历point.link_lines列表，检查是否包含当前line
            if line in valid_points[i].link_lines:
                link_points.append(i)
        if len(link_points) == 2:
            geom_lines.append(
                geom.add_line(geom_points[link_points[0]], geom_points[link_points[1]])
            )
        else:
            raise ValueError(f"交线{line}的连接点数量不为2，实际为{len(link_points)}")

    # 第三步：由线环构建面
    loops = []  # 存储所有有效面环的几何面环对象 和 planes中的面有序对应
    for plane in planes:
        link_lines: list[int] = []
        # 遍历交线列表，找出组成该面的交线
        for i in range(len(cross_lines)):
            if plane == cross_lines[i].plane1 or plane == cross_lines[i].plane2:
                link_lines.append(i)
        # 检查环是否闭合
        open_loop_lines = [geom_lines[i] for i in link_lines]
        close_loop_lines = []
        # 保证环是闭合的
        current_line = open_loop_lines.pop(0)
        close_loop_lines.append(current_line)
        while len(open_loop_lines) > 0:
            # 找到当前线的下一个连接点
            have_next = False
            for j in range(len(open_loop_lines)):
                if current_line.points[1] == open_loop_lines[j].points[0]:
                    current_line = open_loop_lines.pop(j)
                    have_next = True
                    break
                elif current_line.points[1] == open_loop_lines[j].points[1]:
                    current_line = -open_loop_lines.pop(j)
                    have_next = True
                    break
            if not have_next:
                raise ValueError(f"交线{current_line}没有下一个连接点")
            else:
                close_loop_lines.append(current_line)

        loop = geom.add_curve_loop(close_loop_lines)
        face = geom.add_plane_surface(loop)
        loops.append(loop)

    # 第四步：构建闭合面环（核心：面包围体的关键，将6个面组合成封闭面环）
    surf_loop = geom.add_surface_loop(loops)

    # 第五步：面环围合出三维实体（核心操作：add_volume入参为闭合面环）
    volume_3d = geom.add_volume(surf_loop)  # 面包围得到的三维体实体

    # 第六步：生成网格并保存
    mesh = geom.generate_mesh()
    mesh.write("cube_mesh_ascii.vtk", file_format="vtk")

print("三维面包围体网格生成完成，文件：cube_mesh_ascii.vtk")
import pyvista as pv

# 直接读取之前导出的网格可视化
grid = pv.read("cube_mesh_ascii.vtk")
plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=True, color="white")
plotter.add_axes(line_width=2)
# plotter.camera_position = [(2.0, 2.0, 1.0), (0.5, 0.3, 0.0), (0.0, 0.0, 1.0)]
plotter.show()
