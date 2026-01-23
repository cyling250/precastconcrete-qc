import numpy as np
from utils.math_utils import SVD


class LineModel(object):
    def __init__(self):
        # 线段参数（线段上任意一点 cx, cy, cx） + （线段方向向量 nx, ny, nz）
        self.parameters = None

    def calc_inliers(self, points, dst_threshold):
        # 根据点到线的距离计算内点和外点
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        # 计算点到直线的距离
        # 距离公式: |(p - c) × n| / |n| 由于n已经归一化，所以分母为1
        dst = np.linalg.norm(np.cross(points - c, n), axis=1)
        ind = dst < dst_threshold
        return ind

    def estimate_parameters(self, pts, control_line=None):
        # 最小二乘法估算直线模型
        # 只有两个点时，可以直接计算
        num = pts.shape[0]
        if num == 2:
            # 两点确定一条直线
            c = np.mean(pts, axis=0)
            n = pts[1] - pts[0]
            # 归一化方向向量
            n = n / np.linalg.norm(n)
        else:
            # 使用SVD方法拟合直线
            # 对于直线拟合，我们需要找到点云的主方向
            u, s, c, _ = SVD(pts)
            # 主方向是最大特征值对应的特征向量
            n = u[:, 0]
            # 确保方向向量归一化
            n = n / np.linalg.norm(n)

        params = np.hstack((c.reshape(1, -1), n.reshape(1, -1)))[0, :]
        self.parameters = params
        return params


class RansacLineFitter:
    """
    基于RANSAC算法的直线拟合器

    使用RANSAC（随机抽样一致性）算法从三维点云中拟合直线，
    适用于点云数据中的直线结构检测。

    参数:
    threshold: float, 内点距离阈值，默认0.01
    iterations: int, RANSAC迭代次数，默认1000
    min_inliers: int, 最小内点数量，默认10
    random_state: int, 随机种子，默认42
    """

    def __init__(
        self,
        threshold: float = 0.01,
        iterations: int = 1000,
        min_inliers: int = 10,
        random_state: int = 42,
    ):
        self._threshold = threshold
        self._iterations = iterations
        self._min_inliers = min_inliers
        self._random_state = random_state

        # 拟合结果
        self._line_equation = None  # 直线方程参数 [cx, cy, cz, nx, ny, nz]
        self._inliers = None  # 内点索引
        self._outliers = None  # 外点索引

        # 设置随机种子
        np.random.seed(random_state)

    def fit(self, pcd_data: np.array):
        """
        对输入的三维点云进行直线拟合

        参数:
        pcd_data: numpy.ndarray, 形状为 (n_points, 3)，三维点云坐标
        """
        # 验证输入数据
        self._validate_data(pcd_data)

        # 执行RANSAC直线检测
        best_equation, best_inliers = self._ransac_line(pcd_data)

        # 保存拟合结果
        self._line_equation = best_equation
        self._inliers = best_inliers
        self._outliers = np.array(
            [i for i in range(len(pcd_data)) if i not in best_inliers]
        )

    def _validate_data(self, pcd_data: np.ndarray) -> None:
        """
        验证输入数据

        参数:
        pcd_data: numpy.ndarray, 形状为 (n_points, 3)，三维点云坐标
        """
        if pcd_data.ndim != 2 or pcd_data.shape[1] != 3:
            raise ValueError("输入数据形状必须为 (n_points, 3)")

        if len(pcd_data) < 2:
            raise ValueError("至少需要2个点才能拟合直线")

    def _ransac_line(self, pcd_data: np.ndarray):
        """
        RANSAC直线检测核心算法

        参数:
        pcd_data: numpy.ndarray, 三维点云坐标

        返回:
        best_equation: list, 最佳直线方程参数 [cx, cy, cz, nx, ny, nz]
        best_inliers: list, 最佳内点索引列表
        """
        n_points = len(pcd_data)  # 获取点云的点数

        # 如果点太少，直接返回
        if n_points < self._min_inliers:
            return [0, 0, 0, 1, 0, 0], list(range(n_points))

        # 初始化最佳结果
        best_inliers_ratio = 0  # 符合拟合模型的数据的比例
        best_equation = [0, 0, 0, 1, 0, 0]  # 最佳直线方程参数
        best_inliers = []  # 最佳内点索引列表

        # 执行RANSAC迭代
        for i in range(self._iterations):
            # 随机选择n个点
            point_num = np.random.randint(2, n_points / 2)
            sample_index = np.random.choice(n_points, point_num, replace=False)
            sample_points = pcd_data[sample_index, :]

            # 使用LineModel类进行直线拟合
            line = LineModel()
            line_params = line.estimate_parameters(sample_points)

            # 计算内点
            inlier_mask = line.calc_inliers(pcd_data, self._threshold)
            idx_candidates = np.where(inlier_mask)[0]
            inliers_ratio = len(idx_candidates) / n_points

            # 如果当前内点比例大于之前的最佳比例
            if inliers_ratio > best_inliers_ratio:
                best_inliers_ratio = inliers_ratio
                best_equation = line_params.tolist()
                best_inliers = idx_candidates.tolist()

                # 如果找到足够多的内点，提前终止
                if best_inliers_ratio > 0.8:
                    break

        # 检查是否找到足够的内点
        if len(best_inliers) < self._min_inliers:
            print(
                f"警告：只找到 {len(best_inliers)} 个内点，少于最小要求 {self._min_inliers}"
            )

        return best_equation, best_inliers

    def predict(self, pcd_data: np.ndarray) -> np.ndarray:
        """
        预测点云中各点到拟合直线的距离

        参数:
        pcd_data: numpy.ndarray, 形状为 (n_points, 3)，三维点云坐标

        返回:
        numpy.ndarray, 形状为 (n_points,)，各点到直线的距离
        """
        if self._line_equation is None:
            raise ValueError("请先调用fit方法进行直线拟合")

        self._validate_data(pcd_data)

        # 创建直线模型并设置参数
        line = LineModel()
        line.set_parameters(self._line_equation)

        # 计算点到直线的距离
        distances = np.linalg.norm(
            np.cross(
                pcd_data - np.array(self._line_equation[:3]),
                np.array(self._line_equation[3:]),
            ),
            axis=1,
        )

        return distances

    def get_inliers(self) -> np.ndarray:
        """
        获取内点索引

        返回:
        numpy.ndarray, 内点索引列表
        """
        if self._inliers is None:
            raise ValueError("请先调用fit方法进行直线拟合")
        return self._inliers

    def get_outliers(self) -> np.ndarray:
        """
        获取外点索引

        返回:
        numpy.ndarray, 外点索引列表
        """
        if self._outliers is None:
            raise ValueError("请先调用fit方法进行直线拟合")
        return self._outliers

    def get_line_equation(self) -> list:
        """
        获取直线方程参数

        返回:
        list, 直线方程参数 [cx, cy, cz, nx, ny, nz]
        """
        if self._line_equation is None:
            raise ValueError("请先调用fit方法进行直线拟合")
        return self._line_equation

    def get_line_direction(self) -> np.ndarray:
        """
        获取直线方向向量

        返回:
        numpy.ndarray, 直线方向向量 [nx, ny, nz]
        """
        if self._line_equation is None:
            raise ValueError("请先调用fit方法进行直线拟合")
        return np.array(self._line_equation[3:])
