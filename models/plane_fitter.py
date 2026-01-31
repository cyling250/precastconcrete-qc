import numpy as np
import random
from typing import Tuple, List
import time
from tqdm import tqdm
from .plane_model import PlaneModel


class RansacPlaneFitter:
    """
    基于RANSAC算法的平面拟合器

    使用RANSAC（随机抽样一致性）算法从三维点云中拟合平面，
    适用于点云数据中的平面结构检测。

    参数:
    threshold: float, 内点距离阈值，默认0.01
    iterations: int, RANSAC迭代次数，默认100
    min_inliers: int, 最小内点数量，默认10
    max_n_points: int, 最大点数量，默认100
    max_plane_num: int, 最大平面数量，默认1
    random_state: int, 随机种子，默认None
    """

    def __init__(
        self,
        threshold: float = 0.001,
        iterations: int = 1000,
        min_inliers: int = 10,
        max_n_points: int = 4,
        max_plane_num: int = 20,
        random_state: int | None = None,
    ):
        assert max_plane_num > 0, "最大平面数量必须大于0"
        assert type(max_plane_num) == int, "最大平面数量必须为整数"  # noqa: E721

        self._threshold = threshold
        self._iterations = iterations
        self._min_inliers = min_inliers
        self._max_n_points = max_n_points
        self._max_plane_num = max_plane_num
        self._random_state = random_state

        # 拟合结果
        self._plane_equation: List[np.ndarray] = []  # 平面方程参数 [a, b, c, d]
        self._inliers: List[np.ndarray] = []  # 内点坐标列表
        self._outliers: np.ndarray | None = None  # 外点坐标
        self._n_inliers: List[int] = []  # 内点数量

        # 设置随机种子
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def fit(self, pcd_data: np.ndarray):
        """
        对输入的三维点云进行平面拟合，使用RANSAC算法

        参数:
        pcd_data: numpy.ndarray, 形状为 (n_points, 3)，三维点云坐标
        """
        self._outliers = pcd_data.copy()
        # 初始化拟合参数
        n_iter = 0
        while True:
            try:
                self._fit_one(self._outliers)  # 拟合平面
            except ValueError as e:
                print(e)
                break
            # ============停止策略============
            # 1. 找到的平面大于设定的最大平面
            # 2. 剩余内点数量不支持下一次迭代
            # 3. 迭代次数超过最大迭代次数
            # ===============================
            n_iter += 1
            if (
                self._plane_equation is None
                or len(self._plane_equation) >= self._max_plane_num
                or n_iter >= self._iterations
            ):
                break
            if len(self._outliers) < self._min_inliers:
                print(
                    f"第{n_iter}次迭代，剩余外点数量不足{self._min_inliers}个，停止迭代"
                )
                break
        return

    def _fit_one(self, pcd_data: np.ndarray):
        """
        对输入的三维点云进行一次平面拟合，并根据拟合结果更新外点和内点

        参数:
        pcd_data: numpy.ndarray, 形状为 (n_points, 3)，三维点云坐标

        返回:
        best_equation: np.ndarray, 最佳平面方程参数 [a, b, c, d]
        best_inliers: np.ndarray, 最佳内点索引列表
        """
        start_time = time.time()

        # 验证输入数据
        self._validate_data(pcd_data)

        # 执行RANSAC平面检测
        best_equation, best_inliers = self._ransac_plane(pcd_data)

        # 保存拟合结果
        self._plane_equation.append(best_equation)
        self._inliers.append(pcd_data[best_inliers])  # 直接保存内点坐标
        self._n_inliers.append(len(best_inliers))
        # 更新外点
        inlier_mask = np.zeros(len(pcd_data), dtype=bool)
        inlier_mask[best_inliers] = True  # 将内点索引设为True
        self._outliers = pcd_data[~inlier_mask]

        # 计算拟合时间
        fitting_time = time.time() - start_time

        print(
            f"第{len(self._plane_equation)}个平面拟合完成，耗时: {fitting_time:.4f}秒"
        )
        print(f"内点数量: {self._n_inliers[-1]} / {len(pcd_data)}")

        # 平面参数格式: [cx, cy, cz, nx, ny, nz]
        cx, cy, cz, nx, ny, nz = best_equation
        # 计算平面方程的d值: d = - (nx*cx + ny*cy + nz*cz)
        d = -(nx * cx + ny * cy + nz * cz)
        print(
            f"平面方程: {nx:.4f}(x - {cx:.4f}) + {ny:.4f}(y - {cy:.4f}) + {nz:.4f}(z - {cz:.4f}) = 0"
        )
        print(f"标准形式: {nx:.4f}x + {ny:.4f}y + {nz:.4f}z + {d:.4f} = 0")

        return best_equation, best_inliers

    def _validate_data(self, pcd_data: np.ndarray) -> None:
        """验证输入数据"""
        if not isinstance(pcd_data, np.ndarray):
            raise ValueError("输入数据必须是numpy数组")

        if pcd_data.ndim != 2 or pcd_data.shape[1] != 3:
            raise ValueError("输入数据形状必须为 (n_points, 3)")

        if len(pcd_data) < 3:
            raise ValueError("至少需要3个点才能拟合平面")

    def _ransac_plane(self, pcd_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        RANSAC平面检测核心算法

        参数:
        pcd_data: numpy.ndarray, 三维点云坐标

        返回:
        equation: list, 平面方程参数 [a, b, c, d]
        inliers: list, 内点索引列表
        """
        n_points = len(pcd_data)  # 获取点云的点数

        # 如果点太少，直接返回
        if n_points < self._min_inliers:
            return np.array([0, 0, 0, 0]), np.array(range(n_points))

        # 初始化最佳结果
        best_inliers_ratio = 0  # 符合拟合模型的数据的比例
        best_equation = np.array([0, 0, 0, 0])  # 最佳平面方程参数
        best_inliers = np.array([])  # 最佳内点索引列表

        # 使用tqdm显示RANSAC迭代进度
        with tqdm(total=self._iterations, desc="RANSAC平面拟合", unit="iter") as pbar:
            for i in range(self._iterations):
                # 随机选择n个点
                point_num = np.random.randint(3, self._max_n_points)
                sample_index = np.random.choice(n_points, point_num, replace=False)
                sample_points = pcd_data[sample_index, :]

                # 使用PlaneModel类进行平面拟合
                plane = PlaneModel()
                plane_params = plane.estimate_parameters(sample_points)

                # 计算内点
                inlier_mask = plane.calc_inliers(pcd_data, self._threshold)
                idx_candidates = np.where(inlier_mask)[0]
                inliers_ratio = len(idx_candidates) / n_points

                # 如果当前内点比例大于之前的最佳比例
                if inliers_ratio > best_inliers_ratio:
                    best_inliers_ratio = inliers_ratio
                    # 保存plane_model的参数格式 [cx, cy, cz, nx, ny, nz]
                    best_equation = plane_params.tolist()
                    best_inliers = idx_candidates.tolist()

                    # 更新进度条描述，显示当前最佳内点比例
                    pbar.set_postfix(
                        {
                            "最佳内点": len(best_inliers),
                            "内点比例": f"{best_inliers_ratio * 100:.1f}%",
                        }
                    )

                # 如果找到足够多的内点，提前终止
                if best_inliers_ratio > 0.8:
                    pbar.update(self._iterations - i - 1)  # 更新剩余进度
                    break

                pbar.update(1)  # 更新进度条

        # 检查是否找到足够的内点
        if len(best_inliers) < self._min_inliers:
            raise ValueError(
                f"警告：只找到 {len(best_inliers)} 个内点，少于最小要求 {self._min_inliers}"
            )

        return best_equation, best_inliers

    def get_inliers(self) -> List[np.ndarray]:
        """获取内点坐标"""
        if self._inliers is None:
            raise ValueError("请先调用fit方法进行平面拟合")
        return self._inliers

    def set_inliers(self, inliers: List[np.ndarray]) -> None:
        """设置内点坐标"""
        self._inliers = inliers

    def set_outliers(self, outliers: np.ndarray) -> None:
        """设置外点坐标"""
        self._outliers = outliers

    def set_plane_equation(self, plane_equation: List[np.ndarray]) -> None:
        """设置平面方程参数"""
        self._plane_equation = plane_equation

    def get_outliers(self) -> np.ndarray:
        """获取外点坐标"""
        if self._outliers is None:
            raise ValueError("请先调用fit方法进行平面拟合")
        return self._outliers

    def get_plane_equation(self) -> List[np.ndarray]:
        """获取平面方程参数"""
        if self._plane_equation is None:
            raise ValueError("请先调用fit方法进行平面拟合")
        return self._plane_equation

    def get_plane_normal(self) -> np.ndarray:
        """获取平面法向量"""
        if self._plane_equation is None:
            raise ValueError("请先调用fit方法进行平面拟合")
        return np.array(self._plane_equation[:3])
