import open3d as o3d
import numpy as np
from utils.eigen_measures_utils import create_feature_vector


def get_concrete_part(point_cloud, labels, neighbor_features):
    """
    从已经聚类好的点云数据中提取混凝土部分。

    参数:
    point_cloud (numpy.ndarray): 原始点云数据，形状为 (N, 3)，其中 N 是点的数量。
    labels (numpy.ndarray): 点云数据的聚类标签，形状为 (N,)。
    neighbor_features (str): 邻域特征文件的路径。

    返回:
    numpy.ndarray: 混凝土部分的点云数据，形状为 (M, 3)，其中 M 是混凝土点的数量。
    """
    # 读取邻域特征
    feature_vectors, feature_names = create_feature_vector(neighbor_features)
    planarity = feature_vectors[:, feature_names.index("planarity")]
    # 计算不同类别的邻域特征的平均值
    unique_labels = np.unique(labels)
    each_planarity = []
    for label in unique_labels:
        mask = labels == label
        avg_features = np.mean(planarity[mask], axis=0)
        each_planarity.append(avg_features)
        print(f"Cluster {label}: {avg_features}")
    # 最大的平面度的点云是混凝土
    concrete_pcd_array = point_cloud[
        labels == unique_labels[each_planarity.index(max(each_planarity))]
    ]
    concrete_pcd = o3d.geometry.PointCloud()
    concrete_pcd.points = o3d.utility.Vector3dVector(concrete_pcd_array)
    return concrete_pcd


def get_rebar_part(point_cloud, labels, neighbor_features):
    """
    从已经聚类好的点云数据中提取钢筋部分。

    参数:
    point_cloud (numpy.ndarray): 原始点云数据，形状为 (N, 3)，其中 N 是点的数量。
    labels (numpy.ndarray): 点云数据的聚类标签，形状为 (N,)。
    neighbor_features (str): 邻域特征文件的路径。

    返回:
    numpy.ndarray: 钢筋部分的点云数据，形状为 (M, 3)，其中 M 是钢筋点的数量。
    """
    # 读取邻域特征
    feature_vectors, feature_names = create_feature_vector(neighbor_features)
    planarity = feature_vectors[:, feature_names.index("planarity")]
    # 计算不同类别的邻域特征的平均值
    unique_labels = np.unique(labels)
    each_planarity = []
    for label in unique_labels:
        mask = labels == label
        avg_features = np.mean(planarity[mask], axis=0)
        each_planarity.append(avg_features)
        print(f"Cluster {label}: {avg_features}")
    # 最小的平面度的点云是钢筋
    rebar_pcd_array = point_cloud[
        labels == unique_labels[each_planarity.index(min(each_planarity))]
    ]
    rebar_pcd = o3d.geometry.PointCloud()
    rebar_pcd.points = o3d.utility.Vector3dVector(rebar_pcd_array)
    return rebar_pcd
