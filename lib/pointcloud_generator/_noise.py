"""
综合多级噪声增强。

噪声等级按低/中/高递增，每种等级一次性叠加全部四种噪声类型：
  - 块遮挡 (occlusion): 随机矩形块遮蔽部分表面
  - 高斯抖动 (jitter): 沿传感器视线方向扰动点坐标
  - 随机丢弃 (dropout): 随机移除部分点
  - 外来噪点 (outlier): 在包围盒内均匀分布随机点
"""

import numpy as np
from typing import Optional

from ._config import LABEL_NOISE


def apply_noise(
    points: np.ndarray,
    labels: np.ndarray,
    noise_config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    综合多级噪声增强（一次性叠加全部四种噪声）。
    
    Parameters
    ----------
    points : ndarray, (N, 3)
        原始无噪声点云。
    labels : ndarray, (N,)
        原始标签（0=noise, 1=concrete, 2=rebar, 3=sleeve）。
    noise_config : dict
        噪声等级参数，来自 _config.NOISE_LEVELS。
        包含: occlusion_pct, jitter_sigma, keep_pct, outlier_pct。
    rng : np.random.Generator
        
    Returns
    -------
    points_noisy : ndarray, (M, 3)
    labels_noisy : ndarray, (M,)
    """
    occlusion_pct = noise_config.get("occlusion_pct", 0.0)
    jitter_sigma = noise_config.get("jitter_sigma", 0.0)
    keep_pct = noise_config.get("keep_pct", 1.0)
    outlier_pct = noise_config.get("outlier_pct", 0.0)
    
    pts, lbls = points.copy(), labels.copy()
    
    # 1) 块遮挡
    if occlusion_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_occlusion(pts, lbls, occlusion_pct, rng)
    
    # 2) 随机丢弃
    if keep_pct < 1.0 and len(pts) > 0:
        pts, lbls = _apply_dropout(pts, lbls, keep_pct, rng)
    
    # 3) 高斯抖动
    if jitter_sigma > 0 and len(pts) > 0:
        pts = _apply_jitter(pts, jitter_sigma, rng)
    
    # 4) 外来噪点
    if outlier_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_outliers(pts, lbls, outlier_pct, rng)
    
    return pts, lbls


def _apply_occlusion(
    points: np.ndarray,
    labels: np.ndarray,
    occlusion_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    块遮挡：在点云包围盒内随机放置多个矩形块，遮挡内部点被移除。
    遮挡比例 occlusion_pct 表示被遮挡面积占总投影面积的比例。
    """
    if len(points) == 0:
        return points, labels
    
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # 遮挡块数量（1~3 个）
    n_blocks = rng.integers(1, 4)
    mask = np.ones(len(points), dtype=bool)
    
    for _ in range(n_blocks):
        # 随机块大小（覆盖 10%~30% 的包围盒尺寸）
        block_size = bbox_size * rng.uniform(0.1, 0.3, size=3)
        
        # 随机块位置
        block_center = bbox_min + bbox_size * rng.uniform(0.1, 0.9, size=3)
        
        # 查找块内点
        block_min = block_center - block_size / 2
        block_max = block_center + block_size / 2
        
        in_block = (
            (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
            (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]) &
            (points[:, 2] >= block_min[2]) & (points[:, 2] <= block_max[2])
        )
        mask &= ~in_block
    
    return points[mask], labels[mask]


def _apply_dropout(
    points: np.ndarray,
    labels: np.ndarray,
    keep_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    随机丢弃：随机保留 keep_pct 比例的点。
    """
    n = len(points)
    n_keep = max(1, int(n * keep_pct))
    indices = rng.choice(n, size=n_keep, replace=False)
    return points[indices], labels[indices]


def _apply_jitter(
    points: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    高斯抖动：沿每个坐标轴独立添加高斯噪声。
    sigma 为标准差 (mm)。
    """
    noise = rng.normal(0, sigma, size=points.shape)
    return points + noise


def _apply_outliers(
    points: np.ndarray,
    labels: np.ndarray,
    outlier_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    外来噪点：在点云包围盒内均匀分布随机点，标记为噪声 (LABEL_NOISE)。
    """
    n_outliers = max(1, int(len(points) * outlier_pct))
    
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    
    outliers = rng.uniform(
        bbox_min, bbox_max, size=(n_outliers, 3)
    )
    
    points_out = np.vstack([points, outliers])
    labels_out = np.concatenate([
        labels,
        np.full(n_outliers, LABEL_NOISE, dtype=np.int32),
    ])
    
    return points_out, labels_out
