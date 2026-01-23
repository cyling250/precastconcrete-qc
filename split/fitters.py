import numpy as np
from models import RansacPlaneFitter, PlaneModel
from split.get_part import get_concrete_part, get_rebar_part
import open3d as o3d
from scipy.interpolate import UnivariateSpline


def concrete_fit(point_data: np.ndarray, labels: np.ndarray, neighbor_features: str):
    """
    拟合混凝土点云
    """
    concrete_pcd = np.asarray(
        get_concrete_part(point_data, labels, neighbor_features).points
    )

    # 4.2 执行RANSAC
    plane_fitter = RansacPlaneFitter()
    plane_fitter.fit(concrete_pcd)  # 调用RANSAC平面检测函数
    return plane_fitter


def rebar_fit(
    point_data: np.ndarray,
    concrete_plane_fitter: RansacPlaneFitter,
    labels: np.ndarray,
    neighbor_features: str,
):
    # 1 从点云中提取钢筋点
    rebars_points = np.asarray(
        get_rebar_part(point_data, labels, neighbor_features).points
    )
    # 2 在钢筋点云上净化非钢筋点
    concrete_inliers = concrete_plane_fitter.get_inliers()
    concrete_plane_equ = concrete_plane_fitter.get_plane_equation()
    for i in range(len(concrete_inliers)):
        # 计算当前平面与所有点的距离, 筛选出距离小于阈值的点
        plane_model = PlaneModel()
        plane_model.parameters = concrete_plane_equ[i]
        inliners_mask = plane_model.calc_inliers(rebars_points, 0.005)
        # 删除距离大于阈值的点
        rebars_points = rebars_points[~inliners_mask]
    # 3 使用欧式聚类对筛选后的点云进行聚类
    rebars_pcd = o3d.geometry.PointCloud()
    rebars_pcd.points = o3d.utility.Vector3dVector(rebars_points)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(rebars_pcd.cluster_dbscan(eps=0.05, min_points=10))
    # 对每一类的钢筋进行拟合
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        # 筛选当前聚类的点
        cluster_points = rebars_points[labels == cluster_id]
        # plane_fitter = RansacLineFitter()
        # plane_fitter.fit(cluster_points)  # 调用RANSAC平面检测函数
        # inliers = cluster_points[plane_fitter.get_inliers()]
        # outliers = cluster_points[plane_fitter.get_outliers()]

        # # 可视化当前聚类的点云
        # inliers_pcd = o3d.geometry.PointCloud()
        # inliers_pcd.points = o3d.utility.Vector3dVector(inliers)
        # inliers_pcd.colors = o3d.utility.Vector3dVector(
        #     np.tile([1, 0, 0], (len(inliers), 1))
        # )
        # outliers_pcd = o3d.geometry.PointCloud()
        # outliers_pcd.points = o3d.utility.Vector3dVector(outliers)
        # outliers_pcd.colors = o3d.utility.Vector3dVector(
        #     np.tile([0, 0, 1], (len(outliers), 1))
        # )
        # o3d.visualization.draw_geometries([inliers_pcd, outliers_pcd])
