"""
lib/cascade — 级联方法包: S1→S2 的自适应体素降采样管线.
"""

from lib.cascade.filter import filter_patch
from lib.cascade.stage2_model import Stage2Model
from lib.cascade.precompute import build_s2_dataset, load_s2_loader
from lib.cascade.validate import (
    val_v2_simple, val_s2_cached, print_v2, S2_NAMES,
)

__all__ = [
    "filter_patch",
    "Stage2Model",
    "build_s2_dataset", "load_s2_loader",
    "val_v2_simple", "val_s2_cached",
    "print_v2", "S2_NAMES",
]
