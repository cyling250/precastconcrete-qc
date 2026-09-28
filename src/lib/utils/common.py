"""
全局共享常量、设备、种子、标签定义、控制台.

所有模块从此导入公共组件, 避免循环依赖.
"""

import os, random
import numpy as np
import torch
from rich.console import Console

# ── 全局控制台 ──
console = Console()

# ── 设备 ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 超参数默认值 ──
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_POINTS = 8192
BATCH_SIZE = 16
NUM_CLASSES = 4
KEEP_RATIO = 0.4
SAMPLES_PER_EPOCH = 3000

# ── 标签体系 ──
CLASS_NAMES = {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"}
S1_NAMES = {0: "噪声", 1: "混凝土", 2: "金属"}

# ── 项目路径 ──
# 本文件位于 <root>/src/lib/utils/, 四层 dirname 回到项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_BASE = os.path.join(_PROJECT_ROOT, "out")
CHECKPOINT_BASE = os.path.join(OUT_BASE, "checkpoints")
SENSITIVITY_BASE = os.path.join(OUT_BASE, "sensitivity")
MOCK_DIR = os.path.join(_PROJECT_ROOT, "mock", "inuse")
TRAIN_DIR = os.path.join(MOCK_DIR, "train")
VAL_DIR = os.path.join(MOCK_DIR, "val")
TEST_DIR = os.path.join(MOCK_DIR, "test")


def set_seed(seed: int = 42):
    """全局随机种子 (确保可复现)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def map_labels(lbl: torch.Tensor) -> torch.Tensor:
    """4类标签 → S1 3类: 钢筋(2)→金属(2), 套筒(3)→金属(2)."""
    clbl = lbl.clone()
    clbl[clbl == 3] = 2
    return clbl


def serialize_metrics(ious: dict, prf: dict) -> dict:
    """将 IoU 和 PRF 展平为 '指标名→值' 字典 (用于 JSON 保存)."""
    result = {}
    result.update({f"iou/{k}": v for k, v in ious.items()})
    for k, kv in prf.items():
        for m, v in kv.items():
            result[f"prf/{k}/{m}"] = v
    return result
