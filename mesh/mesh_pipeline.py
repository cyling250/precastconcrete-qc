import os
import sys

# 添加项目根目录到Python的模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
import logging
from models import PlaneModel
from mesh.mesh_generator import create_single_plane_mesh
from mesh.mesh_clip import clip_plane_mesh_by_intersection
from mesh.line_in_pcd import line_in_pcd

logging.basicConfig(level=logging.DEBUG)
PCD_NAME = "prefabricated_laminated_beam"


def run_mesh_pipeline():
    """
    运行网格生成和裁剪的完整流程
    """
    # 读取混凝土平面切割结果
    concrete_pcd_all = []
    plane_model_all = []
    for i in range(0, 20):
        concrete_pcd = o3d.io.read_point_cloud(
            f"output/{PCD_NAME}/concrete_inliers_{i}.ply"
        )
        # 读取混凝土平面公式
        plane_params = np.load(f"output/{PCD_NAME}/concrete_plane_params_{i}.npy")
        plane_model = PlaneModel(plane_params)
        plane_model_all.append(plane_model)
        # 将所有点向平面投影，确保所有点都在平面上
        concrete_pcd.points = o3d.utility.Vector3dVector(
            plane_model.project_points(np.asarray(concrete_pcd.points))
        )
        concrete_pcd_all.append(concrete_pcd)

    # 生成单独的网格
    plane_mesh_list = []
    for i in range(len(concrete_pcd_all)):
        # 生成单独的网格
        mesh_tmp = create_single_plane_mesh(
            plane_model_all[i].get_standard(),
            concrete_pcd_all[i],
            color=[0.5, 0.5, 0.5],
        )
        plane_mesh_list.append(mesh_tmp)
    # 保存
    merged_mesh = o3d.geometry.TriangleMesh()
    for mesh in plane_mesh_list:
        merged_mesh += mesh[0]
    # 保存合并后的网格
    output_path = os.path.join(os.getcwd(), "output", "merged_mesh.ply")
    o3d.io.write_triangle_mesh(output_path, merged_mesh)

    # 裁剪网格
    clipped_mesh_list = []
    for i in range(len(plane_mesh_list)):
        # 初始化当前平面的裁剪网格为原始网格
        current_clipped_mesh = plane_mesh_list[i][0]
        for j in range(len(plane_mesh_list)):
            # 跳过自身平面
            if i == j:
                continue
            plane1 = plane_model_all[i]
            plane2 = plane_model_all[j]
            try:
                intersection_line = plane1.intersect(plane2)
            except ValueError:
                continue
            # 判断交线是否在点云内
            if not line_in_pcd(
                intersection_line, plane_model_all[i], concrete_pcd_all[i]
            ):
                continue
            # 网格裁剪
            mesh_tmp = clip_plane_mesh_by_intersection(
                plane1.get_standard(),
                current_clipped_mesh,
                plane_mesh_list[i][1][0],
                plane_mesh_list[i][1][1],
                plane_mesh_list[i][2],
                [intersection_line.parameters[:3], intersection_line.parameters[3:6]],
            )
            # 只有当裁剪成功时才更新当前网格
            if mesh_tmp is not None:
                current_clipped_mesh = mesh_tmp
        # 将最终裁剪后的网格添加到列表中
        clipped_mesh_list.append(current_clipped_mesh)

    # 合并所有裁剪后的平面网格
    merged_mesh = o3d.geometry.TriangleMesh()
    for mesh in clipped_mesh_list:
        merged_mesh += mesh

    # 保存合并后的网格
    output_path = os.path.join(os.getcwd(), "output", "clipped_mesh.ply")
    logging.debug(
        f"合并后的网格包含{len(merged_mesh.vertices)}个顶点，{len(merged_mesh.triangles)}个面"
    )
    logging.debug(f"保存路径: {output_path}")
    try:
        o3d.io.write_triangle_mesh(output_path, merged_mesh)
        logging.debug(f"成功保存裁剪后的网格到 {output_path}")
    except Exception as e:
        logging.error(f"保存网格失败: {e}")


if __name__ == "__main__":
    run_mesh_pipeline()
