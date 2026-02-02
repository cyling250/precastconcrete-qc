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


def point_cloud_intersect(
    pcd1: o3d.geometry.PointCloud | np.ndarray,
    pcd2: o3d.geometry.PointCloud | np.ndarray,
    eps: float = 0.02,  # 距离阈值，核心参数
    use_aabb: bool = True,  # 是否使用AABB包围盒预过滤，默认开启（优化）
) -> bool:
    """
    判断两个点云是否相交（空间重叠）
    :param pcd1: 点云1，Open3D PointCloud对象或numpy数组，形状为(N, 3)
    :param pcd2: 点云2，Open3D PointCloud对象或numpy数组，形状为(M, 3)
    :param eps: 距离阈值，小于该值则认为两点重叠
    :param use_aabb: 是否用AABB包围盒预过滤，大数据量建议开启
    :return: True=相交/重叠，False=不相交
    """

    # 转换numpy数组或Vector3dVector为Open3D PointCloud对象
    def to_point_cloud(pcd):
        if isinstance(pcd, np.ndarray):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pcd)
            return pc
        elif isinstance(pcd, o3d.utility.Vector3dVector):
            pc = o3d.geometry.PointCloud()
            pc.points = pcd
            return pc
        return pcd

    # 转换输入
    pc1 = to_point_cloud(pcd1)
    pc2 = to_point_cloud(pcd2)

    # 优化：AABB包围盒预过滤（新增：先扩张0.01m，再判断相交）
    if use_aabb:
        # 定义包围盒扩张函数（核心：精确各方向扩aabb_expand）
        def expand_aabb(
            aabb: o3d.geometry.AxisAlignedBoundingBox, expand_dist: float
        ) -> o3d.geometry.AxisAlignedBoundingBox:
            """
            将AABB包围盒各方向均匀向外扩张指定距离
            :param aabb: 原始AABB包围盒
            :param expand_dist: 各方向向外扩张的距离（如0.01=每个方向扩0.01m）
            :return: 扩张后的新AABB包围盒
            """
            # 直接调整最小和最大边界值
            min_b = aabb.min_bound  # 原始最小边界 [x, y, z]
            max_b = aabb.max_bound  # 原始最大边界 [x, y, z]

            # 各方向向外扩张指定距离
            new_min = min_b - expand_dist
            new_max = max_b + expand_dist

            # 创建并返回新的AABB包围盒
            return o3d.geometry.AxisAlignedBoundingBox(
                min_bound=new_min, max_bound=new_max
            )

        # 生成原始AABB并扩张0.01m
        aabb1 = expand_aabb(pc1.get_axis_aligned_bounding_box(), eps)
        aabb2 = expand_aabb(pc2.get_axis_aligned_bounding_box(), eps)

        # 手动判断两个AABB包围盒是否相交
        min1, max1 = aabb1.min_bound, aabb1.max_bound
        min2, max2 = aabb2.min_bound, aabb2.max_bound

        # 对于每个维度，检查一个包围盒的最小值是否大于另一个的最大值
        for i in range(3):
            if min1[i] > max2[i] or min2[i] > max1[i]:
                return False  # 包围盒不相交，点云一定不相交

    # 核心：构建KDTree并查询最近邻距离（原有逻辑，无修改）
    if len(pc1.points) > len(pc2.points):
        pcd_query, pcd_tree = pc1, o3d.geometry.KDTreeFlann(pc2)
    else:
        pcd_query, pcd_tree = pc2, o3d.geometry.KDTreeFlann(pc1)

    for point in pcd_query.points:
        # 确保点的形状是(3, 1)
        point = np.asarray(point).reshape(3, 1)
        k, idx, dist_sq = pcd_tree.search_knn_vector_3d(point, knn=1)
        if k > 0 and np.sqrt(dist_sq[0]) < eps:
            return True

    return False
