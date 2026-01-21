import open3d as o3d
import numpy as np


class RegionGrowSplit:
    """
    基于区域生长算法的点云分割类
    使用Open3D实现，参考PCL的RegionGrowing算法
    """

    def __init__(
        self,
        min_cluster_size=50,
        max_cluster_size=1000000,
        number_of_neighbours=30,
        smoothness_threshold=3.0 / 180.0 * np.pi,
        curvature_threshold=1.0,
        k_search=50,
    ):
        """
        初始化区域生长分割参数

        参数:
        min_cluster_size: int, 最小聚类尺寸
        max_cluster_size: int, 最大聚类尺寸
        number_of_neighbours: int, 每个点考虑的邻居数量
        smoothness_threshold: float, 法线角度阈值（弧度）
        curvature_threshold: float, 曲率阈值
        k_search: int, 法线估计时的K近邻数量
        """
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.number_of_neighbours = number_of_neighbours
        self.smoothness_threshold = smoothness_threshold
        self.curvature_threshold = curvature_threshold
        self.k_search = k_search
        self.cloud = None
        self.normals = None
        self.clusters = []

    def load_point_cloud(self, file_path):
        """
        加载点云文件

        参数:
        file_path: str, 点云文件路径

        返回:
        bool, 加载是否成功
        """
        print(f"正在加载点云文件: {file_path}")
        self.cloud = o3d.io.read_point_cloud(file_path)
        if not self.cloud.has_points():
            print("点云读取失败.")
            return False
        print(f"点云加载成功，包含 {len(self.cloud.points)} 个点")
        return True

    def estimate_normals(self):
        """
        估计点云法线

        返回:
        bool, 法线估计是否成功
        """
        if not self.cloud:
            print("请先加载点云")
            return False

        print("正在估计法线...")
        # 使用K近邻搜索估计法线
        self.cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(self.k_search)
        )
        print("法线估计完成")
        return True

    def apply_pass_through_filter(self, z_min=0.0, z_max=1.0):
        """
        应用直通滤波器，过滤z轴范围内的点

        参数:
        z_min: float, z轴最小值
        z_max: float, z轴最大值

        返回:
        bool, 滤波是否成功
        """
        if not self.cloud:
            print("请先加载点云")
            return False

        print(f"应用直通滤波器，z范围: [{z_min}, {z_max}]")
        # 转换为numpy数组进行过滤
        points = np.asarray(self.cloud.points)
        # 过滤z值在范围内的点
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        filtered_points = points[mask]

        # 创建新的点云对象
        filtered_cloud = o3d.geometry.PointCloud()
        filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        # 如果已估计法线，也过滤法线
        if self.cloud.has_normals():
            normals = np.asarray(self.cloud.normals)
            filtered_normals = normals[mask]
            filtered_cloud.normals = o3d.utility.Vector3dVector(filtered_normals)

        self.cloud = filtered_cloud
        print(f"滤波完成，剩余 {len(self.cloud.points)} 个点")
        return True

    def segment(self):
        """
        执行区域生长分割

        返回:
        bool, 分割是否成功
        """
        if not self.cloud:
            print("请先加载点云")
            return False

        if not self.cloud.has_normals():
            print("请先估计法线")
            return False

        print("正在执行区域生长分割...")

        # 创建KDTree用于最近邻搜索
        kdtree = o3d.geometry.KDTreeFlann(self.cloud)

        # 初始化标签
        labels = np.full(len(self.cloud.points), -1, dtype=int)
        cluster_id = 0

        # 遍历所有点
        for i in range(len(self.cloud.points)):
            # 如果点已经被标记，跳过
            if labels[i] != -1:
                continue

            # 初始化当前聚类
            seed_queue = [i]
            current_cluster = []

            # 从种子点开始生长
            while seed_queue:
                current_idx = seed_queue.pop(0)

                # 如果点已经被标记，跳过
                if labels[current_idx] != -1:
                    continue

                # 标记当前点
                labels[current_idx] = cluster_id
                current_cluster.append(current_idx)

                # 查找近邻点
                [_, idx, _] = kdtree.search_knn_vector_3d(
                    self.cloud.points[current_idx], self.number_of_neighbours
                )

                # 遍历近邻点
                for j in idx:
                    if j == current_idx or labels[j] != -1:
                        continue

                    # 计算法线角度差
                    normal1 = self.cloud.normals[current_idx]
                    normal2 = self.cloud.normals[j]
                    angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))

                    # 检查法线角度是否在阈值内
                    if angle < self.smoothness_threshold:
                        seed_queue.append(j)

            # 检查聚类大小
            if self.min_cluster_size <= len(current_cluster) <= self.max_cluster_size:
                self.clusters.append(current_cluster)
                cluster_id += 1

        print(f"分割完成，得到 {len(self.clusters)} 个聚类")

        # 输出每个聚类的点数量
        for i, cluster in enumerate(self.clusters):
            print(f"聚类 {i} 包含 {len(cluster)} 个点")

        return True

    def visualize(self):
        """
        可视化分割结果
        """
        if not self.cloud or not self.clusters:
            print("请先执行分割")
            return

        print("正在可视化分割结果...")

        # 创建彩色点云
        colored_cloud = o3d.geometry.PointCloud()
        colored_cloud.points = self.cloud.points

        # 为每个聚类分配不同颜色
        colors = np.zeros((len(self.cloud.points), 3))
        for i, cluster_indices in enumerate(self.clusters):
            # 生成随机颜色
            color = np.random.rand(3)
            colors[cluster_indices] = color

        colored_cloud.colors = o3d.utility.Vector3dVector(colors)

        # 可视化
        o3d.visualization.draw_geometries(
            [colored_cloud], window_name="区域生长分割结果"
        )

    def run(self, file_path, z_min=0.0, z_max=1.0):
        """
        运行完整的区域生长分割流程

        参数:
        file_path: str, 点云文件路径
        z_min: float, 直通滤波器z轴最小值
        z_max: float, 直通滤波器z轴最大值

        返回:
        bool, 流程是否成功完成
        """
        # 1. 加载点云
        if not self.load_point_cloud(file_path):
            return False

        # 2. 估计法线
        if not self.estimate_normals():
            return False

        # 3. 应用直通滤波器
        if not self.apply_pass_through_filter(z_min, z_max):
            return False

        # 4. 执行分割
        if not self.segment():
            return False

        # 5. 可视化结果
        self.visualize()

        return True


# 示例用法
if __name__ == "__main__":
    # 创建区域生长分割对象
    segmenter = RegionGrowSplit()

    # 运行分割流程
    # 注意：需要替换为实际的点云文件路径
    file_path = "region_growing_tutorial.ply"
    segmenter.run(file_path)
