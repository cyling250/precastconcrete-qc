import open3d as o3d
import numpy as np

from models import PlaneModel

PCD_NAME = "prefabricated_column"

print("Testing mesh in Open3D...")
# 读取混凝土平面切割结果
concrete_pcd_all = []
for i in range(0, 6):
    concrete_pcd = o3d.io.read_point_cloud(
        f"output/{PCD_NAME}/concrete_inliers_{i}.ply"
    )
    # 读取混凝土平面公式
    plane_params = np.load(f"output/{PCD_NAME}/concrete_plane_params_{i}.npy")
    plane_model = PlaneModel(plane_params)
    # 将所有点向平面投影，确保所有点都在平面上
    concrete_pcd.points = o3d.utility.Vector3dVector(
        plane_model.project_points(np.asarray(concrete_pcd.points))
    )
    concrete_pcd_all.append(concrete_pcd)
concrete_pcd = o3d.geometry.PointCloud()
concrete_pcd.points = o3d.utility.Vector3dVector(
    np.concatenate([np.asarray(pcd.points) for pcd in concrete_pcd_all], axis=0)
)
# 法向量估计
concrete_pcd.paint_uniform_color([1.0, 0.0, 0.0])
concrete_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
)
distances = concrete_pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    concrete_pcd, o3d.utility.DoubleVector([radius, radius * 2])
)
bbox = concrete_pcd.get_axis_aligned_bounding_box()
mesh_crop = mesh.crop(bbox)
# 可视化点云
o3d.io.write_triangle_mesh(f"output/{PCD_NAME}/mesh_c.ply", mesh_crop)
