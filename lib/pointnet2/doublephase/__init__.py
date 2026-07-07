"""
Cascade pipeline utilities for the two-phase segmentation approach.

Phase 1 (S1): 3-class (noise/concrete/metal)
Phase 2 (S2): 4-class refinement (noise/concrete/rebar/sleeve)
               using adaptive voxel downsampling on concrete.
"""

from .cascade_utils import (
    Stage2Model,
    filter_patch,
    build_s2_dataset,
    load_s2_loader,
    val_v2_simple,
    val_s2_cached,
    print_v2,
    S2_NAMES,
)

__all__ = [
    "Stage2Model",
    "filter_patch",
    "build_s2_dataset",
    "load_s2_loader",
    "val_v2_simple",
    "val_s2_cached",
    "print_v2",
    "S2_NAMES",
]
