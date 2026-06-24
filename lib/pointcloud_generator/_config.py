"""
全局配置：标签体系、默认参数、随机状态。
"""

import random
import numpy as np
from typing import Optional

# ─── 标签体系 ───────────────────────────────────────────────
LABEL_NOISE: int = 0
LABEL_CONCRETE: int = 1
LABEL_REBAR: int = 2
LABEL_SLEEVE: int = 3
NUM_CLASSES: int = 4
CLASS_NAMES: list[str] = ["noise", "concrete", "rebar", "sleeve"]

# ─── 随机种子管理 ──────────────────────────────────────────
_SEED: int | None = None


def set_seed(seed: int) -> None:
    global _SEED
    _SEED = seed
    random.seed(seed)
    np.random.seed(seed)


def get_rng():
    return random.Random(_SEED) if _SEED is not None else random


# ─── 数据目录 ──────────────────────────────────────────────
MOCK_DIR: str = "mock"
ORIGIN_DIR: str = "mock/origin"
ADD_NOISE_DIR: str = "mock/add_noise"
INUSE_DIR: str = "mock/inuse"
TRAIN_DIR: str = "mock/inuse/train"
VAL_DIR: str = "mock/inuse/val"
TEST_DIR: str = "mock/inuse/test"

# ─── 采样密度基准 ──────────────────────────────────────────
# 目标：每样本 ~200K 点（训练时随机采样至 8192 点）
DENSITY_HIGH: float = 0.02
DENSITY_MEDIUM: float = 0.012
DENSITY_LOW: float = 0.006

DENSITY_MAP: dict[str, float] = {
    "high": DENSITY_HIGH,
    "medium": DENSITY_MEDIUM,
    "low": DENSITY_LOW,
}

# ─── 套筒默认参数范围 (JG/T 398-2019) ─────────────────────
SLEEVE_OUTER_DIAM_RANGE: tuple[float, float] = (36.0, 75.0)
SLEEVE_INNER_DIAM_RANGE: tuple[float, float] = (25.0, 55.0)
SLEEVE_DEPTH_RANGE: tuple[float, float] = (140.0, 400.0)
SLEEVE_WALL_THICKNESS: float = 6.0  # 壁厚 6~12 mm
SLEEVE_INNER_DENSITY_FACTOR: float = 0.5  # 内壁密度 = 基准 × 0.5
SLEEVE_BOTTOM_DENSITY_FACTOR: float = 5.0  # 底部密度 = 基准 × 5.0 (确保底部可见)

# ─── 钢筋默认参数 (GB 50010, JGJ 1-2014) ──────────────
REBAR_DIAM_RANGE: tuple[float, float] = (16.0, 32.0)  # 备用连续范围（推荐使用标准系列）
CONCRETE_COVER: float = 25.0  # 纵筋保护层厚度 mm (混凝土表面→钢筋中心)
STIRRUP_DIAM_RANGE: tuple[float, float] = (8.0, 12.0)  # 预制构件箍筋直径 mm (非标不可用标准系列)
STIRRUP_SPACING_RANGE: tuple[float, float] = (100.0, 250.0)
SLEEVE_MIN_COVER: float = 20.0  # 套筒外壁最小保护层厚度 mm (JGJ 1-2014 6.5.3)

# 钢筋公称直径标准系列 (GB 1499.2-2024)
# 热轧带肋钢筋标准直径：6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 50
# 预制构件常用范围 16~32mm
STANDARD_REBAR_DIAMETERS: list[float] = [16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 32.0]

# 灌浆套筒规格映射 (JG/T 398-2019, 全灌浆套筒)
# {钢筋直径: (套筒外径, 套筒内腔直径, 套筒长度)}
# 数据综合利物宝GT系列、GTQ4J钢制套筒等多厂商产品规格
# 套筒内径已满足 JGJ 355-2015 §3.1.2 要求（d≤25时≥d+10, d≥28时≥d+15）
SLEEVE_SPEC_MAP: dict[float, tuple[float, float, float]] = {
    16.0: (48.0, 36.0, 310.0),
    18.0: (50.0, 38.0, 350.0),
    20.0: (52.0, 40.0, 370.0),
    22.0: (54.0, 42.0, 410.0),
    25.0: (58.0, 46.0, 460.0),
    28.0: (62.0, 50.0, 505.0),
    32.0: (66.0, 54.0, 570.0),
}

# ─── 噪声默认参数 ──────────────────────────────────────────
NOISE_LEVELS: dict[str, dict] = {
    "low": {
        "occlusion_pct": 0.20,
        "jitter_sigma": 2.0,
        "keep_pct": 0.85,
        "outlier_pct": 0.01,
    },
    "medium": {
        "occlusion_pct": 0.30,
        "jitter_sigma": 5.0,
        "keep_pct": 0.75,
        "outlier_pct": 0.03,
    },
    "high": {
        "occlusion_pct": 0.40,
        "jitter_sigma": 8.0,
        "keep_pct": 0.65,
        "outlier_pct": 0.05,
    },
}

# ─── 构件几何参数范围 ──────────────────────────────────
COLUMN_WIDTH_RANGE: tuple[float, float] = (400.0, 800.0)
COLUMN_DEPTH_RANGE: tuple[float, float] = (400.0, 800.0)
COLUMN_HEIGHT_RANGE: tuple[float, float] = (2000.0, 6000.0)
L_SHAPED_COLUMN_FLANGE_RANGE: tuple[float, float] = (400.0, 700.0)
L_SHAPED_COLUMN_WEB_RANGE: tuple[float, float] = (300.0, 500.0)

BEAM_WIDTH_RANGE: tuple[float, float] = (300.0, 600.0)
BEAM_HEIGHT_RANGE: tuple[float, float] = (400.0, 800.0)
BEAM_LENGTH_RANGE: tuple[float, float] = (3000.0, 9000.0)

T_BEAM_FLANGE_WIDTH_RANGE: tuple[float, float] = (500.0, 1000.0)
T_BEAM_FLANGE_HEIGHT_RANGE: tuple[float, float] = (100.0, 200.0)
T_BEAM_WEB_WIDTH_RANGE: tuple[float, float] = (200.0, 400.0)
T_BEAM_TOTAL_HEIGHT_RANGE: tuple[float, float] = (500.0, 1000.0)

# ─── 钢筋排布参数 ──────────────────────────────────────
NUM_REBAR_CORNERS: int = 4  # 角部纵筋
SPACING_ALONG_EDGE: float = 100.0  # 沿边分布间距 (mm)

# ─── 采样配置 ──────────────────────────────────────────────
CONCRETE_ROUGHNESS: float = 0.4  # 混凝土表面粗糙度 σ (mm)，沿法线方向高斯扰动
CONCRETE_POINTS_PER_AREA: int = 5  # 每 mm² 点数基准
REBAR_POINTS_PER_LENGTH: int = 200  # 每 mm 点数基准
NUM_SLEEVE_CIRCUMFERENTIAL: int = 8  # 套筒周向采样点数

# ─── 文件名配置 ──────────────────────────────────────────
FILE_PREFIX: str = "Mock"
NUM_VARIANTS_PER_TYPE: int = 10
TRAIN_VAL_TEST_SPLIT: tuple[int, int, int] = (6, 2, 2)  # 6:2:2
