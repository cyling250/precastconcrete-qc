import numpy as np
import open3d as o3d
import logging
from shapely.geometry import Polygon, LineString, Point, MultiPolygon


def clip_plane_mesh_by_intersection(plane, mesh, u, v, centroid, intersect_line):
    """
    用平面交线裁剪面网格，保留实体内侧有效区域
    :param plane: 标准化平面方程
    :param mesh: 待裁剪的初始面网格
    :param u,v: 平面局部二维坐标系
    :param centroid: 平面中心
    :param intersect_line: 平面与另一平面的交线(P0, v)
    :return: 裁剪后的面网格（TriangleMesh），无有效区域则返回None
    """
    A, B, C, D = plane
    P0_line, v_line = intersect_line
    n = np.array([A, B, C])

    # 步骤1：将面网格顶点投影到二维局部坐标系
    verts_3d = np.asarray(mesh.vertices)
    verts_centered = verts_3d - centroid
    verts_2d = np.column_stack([np.dot(verts_centered, u), np.dot(verts_centered, v)])
    # 生成初始二维多边形（面网格轮廓）
    poly_2d = Polygon(verts_2d)

    # 步骤2：将交线投影到二维局部坐标系，生成裁剪线
    # 取交线上2个点（沿方向向量延伸，覆盖面网格范围）
    line_p1 = P0_line + 10 * np.linalg.norm(verts_3d - centroid) * v_line
    line_p2 = P0_line - 10 * np.linalg.norm(verts_3d - centroid) * v_line
    # 交线两点投影到二维
    line_p1_centered = line_p1 - centroid
    line_p2_centered = line_p2 - centroid
    line_p1_2d = np.array([np.dot(line_p1_centered, u), np.dot(line_p1_centered, v)])
    line_p2_2d = np.array([np.dot(line_p2_centered, u), np.dot(line_p2_centered, v)])
    clip_line = LineString([line_p1_2d, line_p2_2d])

    # 步骤3：布尔裁剪（保留实体内侧的多边形部分）
    # 判断裁剪方向：取平面中心向实体内侧的点，确定保留区域
    inner_point_3d = centroid - 0.1 * n  # 实体内侧点（法向量朝外，故减法）
    inner_point_centered = inner_point_3d - centroid
    inner_point_2d = np.array(
        [np.dot(inner_point_centered, u), np.dot(inner_point_centered, v)]
    )
    inner_point_shapely = Point(inner_point_2d)

    # 裁剪：保留包含内侧点的多边形部分
    if poly_2d.intersects(clip_line):
        clipped_poly = poly_2d.difference(Polygon(clip_line.buffer(1e-6)))
        # 若裁剪后无有效区域，返回None
        if clipped_poly.is_empty:
            logging.debug("裁剪后多边形为空")
            return None

        # 处理MultiPolygon情况，选择包含内侧点的多边形
        if type(clipped_poly) is MultiPolygon:
            logging.debug(
                f"裁剪后为MultiPolygon，包含{len(clipped_poly.geoms)}个多边形"
            )
            # 遍历所有多边形，找到包含内侧点的那个
            for i, geom in enumerate(clipped_poly.geoms):
                if geom.contains(inner_point_shapely):
                    logging.debug(f"选择第{i}个多边形，包含内侧点")
                    clipped_poly = geom
                    break
            else:
                # 没有找到包含内侧点的多边形，返回None
                logging.debug("没有找到包含内侧点的多边形")
                return None

        # 确保现在是单一Polygon
        if type(clipped_poly) is not Polygon:
            logging.debug(f"裁剪后不是单一Polygon，而是{type(clipped_poly)}")
            return None

        # 提取裁剪后的二维顶点
        clipped_2d = np.array(clipped_poly.exterior.coords)[:-1]  # 剔除重复的最后一个点
        logging.debug(f"裁剪后顶点数: {len(clipped_2d)}")
    else:
        logging.debug("多边形与裁剪线无相交，保留原轮廓")
        clipped_2d = verts_2d  # 无相交，保留原轮廓

    # 步骤4：裁剪后的二维顶点还原为三维坐标
    clipped_3d = centroid + clipped_2d[:, 0:1] * u + clipped_2d[:, 1:2] * v

    # 步骤5：重新三角化生成裁剪后的面网格
    if len(clipped_3d) < 3:
        logging.debug(f"裁剪后顶点数不足3个，无法生成网格")
        return None  # 顶点数不足，无法生成网格
    # 三角化（简单多边形→三角形，适配凸多边形，装配式RC构件均满足）
    triangles = []
    for i in range(2, len(clipped_3d)):
        triangles.append([0, i - 1, i])
    triangles = np.array(triangles)
    clipped_mesh = o3d.geometry.TriangleMesh()
    clipped_mesh.vertices = o3d.utility.Vector3dVector(clipped_3d)
    clipped_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    clipped_mesh.paint_uniform_color(np.asarray(mesh.vertex_colors)[0])
    clipped_mesh.compute_vertex_normals()

    logging.debug(
        f"裁剪成功，生成网格，顶点数: {len(clipped_mesh.vertices)}, 面数: {len(clipped_mesh.triangles)}"
    )
    return clipped_mesh