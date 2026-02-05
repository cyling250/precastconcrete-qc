import time
from typing import List
import numpy as np
from tqdm import tqdm
import open3d as o3d
from geometry import BaseLine, estimate_line, calc_inliers_line
import logging


class RansacLineFitter:
    """
    基于RANSAC算法的直线拟合器

    使用RANSAC（随机抽样一致性）算法从三维点云中拟合直线，
    适用于点云数据中的直线结构检测。

    参数:
    threshold: float, 内点距离阈值，默认0.01
    iterations: int, RANSAC迭代次数，默认1000
    min_inliers: int, 最小内点数量，默认10
    max_n_points: int, 最大点数量，默认10
    random_state: int, 随机种子，默认42
    """

    def __init__(
        self,
        threshold: float = 0.006,
        iterations: int = 5000,
        min_inliers: int = 10,
        max_n_points: int = 10,
        random_state: int = 42,
        linearity_threshold: float = 0.6,
        idx: int = 0,
    ):
        self._threshold = threshold
        self._iterations = iterations
        self._min_inliers = min_inliers
        self._max_n_points = max_n_points
        self._random_state = random_state
        self._linearity_threshold = linearity_threshold
        self._idx = idx

        # 拟合结果
        # 直线方程参数 [cx, cy, cz, nx, ny, nz]
        self._line: List[BaseLine] = []
        self._inliers: List[np.ndarray] = []  # 内点坐标列表
        self._n_inliers: List[int] = []  # 内点数量
        self._outliers: np.ndarray | None = None  # 外点坐标

        # 设置随机种子
        np.random.seed(random_state)

    def fit(self, pcd_data: np.ndarray):
        """
        对输入的三维点云进行RANSAC直线拟合

        参数:
        pcd_data: numpy.ndarray, 形状为 (n_points, 3)，三维点云坐标
        """
        self._outliers = pcd_data.copy()
        # 初始化拟合参数
        n_iter = 0
        self._fit_one(self._outliers)  # 拟合直线
        if self._outliers.shape[0] < self._min_inliers:
            return
        while True:
            try:
                self._fit_one(self._outliers, self._line)  # 拟合直线
            except ValueError as e:
                print(e)
                break
            # ============停止策略============
            # 1. 找到的直线大于设定的最大直线数
            # 2. 剩余内点数量不支持下一次迭代
            # 3. 迭代次数超过最大迭代次数
            # ===============================
            n_iter += 1
            if (
                self._line is not None
                and len(self._line) >= self._max_n_points
                or n_iter >= self._iterations
            ):
                break
            if len(self._outliers) < self._min_inliers:
                logging.debug(
                    f"第{n_iter}次迭代，剩余外点数量不足{self._min_inliers}个，停止迭代"
                )
                break
        return

    def _fit_one(
        self, pcd_data: np.ndarray, control_lines: List[np.ndarray] | None = None
    ):
        """
        对输入的三维点云进行直线拟合

        参数:
        pcd_data: numpy.ndarray, 形状为 (n_points, 3)，三维点云坐标
        control_lines: Optional[List[np.ndarray]], 可选参数，当提供该参数时，所拟合的直线将会与该直线相交，每个直线为 [cx, cy, cz, nx, ny, nz] 格式
        """
        # 记录开始时间
        start_time = time.time()

        # 验证输入数据
        self._validate_data(pcd_data)

        # 基于半径的离群点剔除
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data)
        cl, ind = pcd.remove_radius_outlier(
            nb_points=3, radius=0.05
        )  # 5cm范围内至少有3个点
        pcd_data = np.asarray(pcd.points)[ind]

        # 执行RANSAC直线检测
        best_line, best_inliers = self._ransac_line(pcd_data, control_lines)

        # 保存拟合结果
        self._line.append(best_line)
        self._inliers.append(pcd_data[best_inliers])  # 直接保存内点坐标
        self._n_inliers.append(len(best_inliers))
        # 更新外点
        inlier_mask = np.zeros(len(pcd_data), dtype=bool)
        inlier_mask[best_inliers] = True  # 将内点索引设为True
        self._outliers = pcd_data[~inlier_mask]
        # 计算拟合时间
        fitting_time = time.time() - start_time
        logging.debug(f"第{len(self._line)}个直线拟合完成，耗时: {fitting_time:.4f}秒")
        logging.debug(f"内点数量: {self._n_inliers[-1]} / {len(pcd_data)}")
        # 直线参数格式: [cx, cy, cz, nx, ny, nz]
        cx, cy, cz, nx, ny, nz = best_line.parameters
        logging.debug(
            f"直线方程: (x - {cx:.4f})/{nx:.4f} = (y - {cy:.4f})/{ny:.4f} = (z - {cz:.4f})/{nz:.4f}"
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

    def _ransac_line(
        self, pcd_data: np.ndarray, control_lines: List[np.ndarray] | None = None
    ) -> [BaseLine, np.ndarray]:
        """
        RANSAC直线检测核心算法

        参数:
        pcd_data: numpy.ndarray, 三维点云坐标
        control_lines: Optional[List[np.ndarray]] = None, 可选参数，当提供该参数时，所拟合的直线将会与该直线相交，该直线为 [cx, cy, cz, nx, ny, nz] 格式

        返回:
        best_equation: list, 最佳直线方程参数 [cx, cy, cz, nx, ny, nz]
        best_inliers: list, 最佳内点索引列表
        """
        n_points = len(pcd_data)  # 获取点云的点数

        # 如果点太少，直接返回
        if n_points < self._min_inliers:
            raise ValueError(
                f"警告：点云数据数量 {n_points} 少于最小要求 {self._min_inliers} 个，无法拟合直线"
            )

        # 初始化最佳结果
        best_inliers_ratio = 0  # 符合拟合模型的数据的比例
        best_line: BaseLine = None  # 最佳直线方程参数
        best_inliers = []  # 最佳内点索引列表

        # 使用tqdm显示RANSAC迭代进度
        with tqdm(
            total=self._iterations,
            desc=f"直线拟合<{self._idx}-{len(self._line)}>",
            unit="iter",
        ) as pbar:
            for i in range(self._iterations):
                # 随机选择n个点
                point_num = np.random.randint(2, self._max_n_points)
                sample_index = np.random.choice(n_points, point_num, replace=False)
                sample_points = pcd_data[sample_index, :]

                # 从所给控制直线中随机选择一条
                if control_lines is not None:
                    control_line = control_lines[
                        np.random.randint(0, len(control_lines))
                    ]
                else:
                    control_line = None
                line_tmp = estimate_line(sample_points, control_line)

                # 计算内点
                inlier_mask = calc_inliers_line(line_tmp, pcd_data, self._threshold)
                idx_candidates = np.where(inlier_mask)[0]
                inliers_ratio = len(idx_candidates) / n_points

                # 如果当前内点比例大于之前的最佳比例
                if inliers_ratio > best_inliers_ratio:
                    best_inliers_ratio = inliers_ratio
                    best_line = line_tmp
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
                    pbar.update(self._iterations - i)  # 更新剩余进度
                    break

                pbar.update(1)  # 更新进度条

        # 检查是否找到足够的内点
        if len(best_inliers) < self._min_inliers:
            raise ValueError(
                f"警告：只找到 {len(best_inliers)} 个内点，少于最小要求 {self._min_inliers}"
            )

        return best_line, np.array(best_inliers)

    def get_inliers(self) -> List[np.ndarray]:
        """
        获取内点索引

        返回:
        List[numpy.ndarray], 内点索引列表
        """
        if len(self._inliers) == 0:
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

    def get_lines(self) -> list:
        """
        获取直线方程参数

        返回:
        list, 直线方程参数 [[cx, cy, cz, nx, ny, nz], ...]
        """
        if len(self._line) == 0:
            raise ValueError("请先调用fit方法进行直线拟合")
        return self._line
