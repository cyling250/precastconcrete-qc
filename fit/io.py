import open3d as o3d
import numpy as np
from models import RansacPlaneFitter, RansacLineFitter


def save_plane_params(plane_fitter: RansacPlaneFitter, output_dir: str, pcd_name: str):
    """
    保存平面参数到文件

    Args:
        plane_fitter (RansacPlaneFitter): 平面拟合器对象
        output_dir (str): 输出目录
        pcd_name (str): 点云文件名
    """
    all_point = plane_fitter.get_inliers()
    pcd_list = []
    for i in range(len(all_point)):
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(all_point[i])

        # 点云的体积选取合适的体素大小，当点云数量少于500个时，不进行下采样
        if len(all_point[i]) > 500:
            pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.01)
            all_point[i] = np.asarray(pcd_temp.points)

        # 离群点删除
        _, ind = pcd_temp.remove_radius_outlier(nb_points=1, radius=0.04)
        pcd_temp.points = o3d.utility.Vector3dVector(np.asarray(pcd_temp.points)[ind])
        # 为每个点云生成不同的随机颜色
        random_color = np.random.rand(3)  # 生成随机RGB颜色
        current_points = np.asarray(pcd_temp.points)
        pcd_temp.colors = o3d.utility.Vector3dVector(
            np.tile(random_color, (len(current_points), 1))
        )
        # 保存平面点云
        o3d.io.write_point_cloud(
            f"{output_dir}/{pcd_name}/concrete_inliers_{i}.ply", pcd_temp
        )
        # 保存平面公式
        plane_params = plane_fitter.get_plane()[i].parameters
        np.save(f"{output_dir}/{pcd_name}/concrete_plane_params_{i}.npy", plane_params)
        pcd_list.append(pcd_temp)
    return pcd_list


def save_line_params(
    line_fitters: list[RansacLineFitter], output_dir: str, pcd_name: str
):
    """
    保存线参数到文件

    Args:
        line_fitters (list[RansacLineFitter]): 线拟合器对象列表
        output_dir (str): 输出目录
        pcd_name (str): 点云文件名
    """
    # 保存钢筋拟合结果
    pcd_list = []
    for i, line_fitter in enumerate(line_fitters):
        inliers_list = line_fitter.get_inliers()
        for j, inliers in enumerate(inliers_list):
            # 保存点云数据为ply
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(inliers)
            o3d.io.write_point_cloud(
                f"{output_dir}/{pcd_name}/rebar_inliers_{i}_{j}.ply", pcd_temp
            )
            # 为每个点云生成不同的随机颜色
            random_color = np.random.rand(3)  # 生成随机RGB颜色
            current_points = np.asarray(pcd_temp.points)
            pcd_temp.colors = o3d.utility.Vector3dVector(
                np.tile(random_color, (len(current_points), 1))
            )
            # 保存直线公式
            line_params = line_fitter.get_lines()[j].parameters
            np.save(
                f"{output_dir}/{pcd_name}/rebar_line_params_{i}_{j}.npy", line_params
            )
            # 添加原始内点到列表中
            pcd_list.append(pcd_temp)
    return pcd_list
