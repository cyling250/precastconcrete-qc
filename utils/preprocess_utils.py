import open3d as o3d
import numpy as np


def desample_point_cloud(
    pcd: o3d.geometry.PointCloud, voxel_size: float = 0.05
) -> o3d.geometry.PointCloud:
    """
    对点云进行体素下采样。

    参数:
    pcd (o3d.geometry.PointCloud): 输入的点云对象。
    voxel_size (float): 体素大小，默认值为0.005。

    返回:
    o3d.geometry.PointCloud: 下采样后的点云对象。
    """
    return pcd.voxel_down_sample(voxel_size)
