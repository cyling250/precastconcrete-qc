"""
预制混凝土构件点云生成器。

依据 GB 50010-2010 / JG/T 398-2019 等规范，参数化生成七种装配式构件的
三维点云，支持混凝土 / 钢筋 / 灌浆套筒三类材料标签和综合多级噪声增强。

公开 API:
  make_point_cloud(...)       — 单构件点云生成入口
  build_rectangular_column(...) — 矩形柱
  build_l_shaped_column(...)    — L 形柱
  build_rectangular_beam(...)   — 矩形梁
  build_t_beam(...)             — T 形梁
  build_rectangular_composite_beam(...) — 矩形叠合梁
  build_t_composite_beam(...)         — T 形叠合梁
  build_l_composite_beam(...)         — L 形叠合梁
  DatasetGenerator(...)         — 批量数据集生成器

标签体系:
  0 = noise (噪声/背景)
  1 = concrete (混凝土)
  2 = rebar (钢筋)
  3 = sleeve (灌浆套筒)
"""

from ._config import (
    LABEL_NOISE, LABEL_CONCRETE, LABEL_REBAR, LABEL_SLEEVE,
    NUM_CLASSES, CLASS_NAMES,
    set_seed, get_rng,
)
from ._noise import apply_noise
from ._concrete import sample_concrete_prism, remove_circular_hole
from ._rebar import generate_longitudinal_bars, generate_stirrups_rect, generate_stirrups_polygon
from ._sleeve import generate_sleeve, generate_sleeves_for_component
from ._builder import (
    build_rectangular_column,
    build_l_shaped_column,
    build_rectangular_beam,
    build_t_beam,
    build_rectangular_composite_beam,
    build_t_composite_beam,
    build_l_composite_beam,
    make_point_cloud,
)
from ._dataset import DatasetGenerator

__all__ = [
    # 标签
    "LABEL_NOISE", "LABEL_CONCRETE", "LABEL_REBAR", "LABEL_SLEEVE",
    "NUM_CLASSES", "CLASS_NAMES",
    # 种子
    "set_seed", "get_rng",
    # 噪声
    "apply_noise",
    # 混凝土
    "sample_concrete_prism", "remove_circular_hole",
    # 钢筋
    "generate_longitudinal_bars", "generate_stirrups_rect", "generate_stirrups_polygon",
    # 套筒
    "generate_sleeve", "generate_sleeves_for_component",
    # 构件
    "build_rectangular_column", "build_l_shaped_column",
    "build_rectangular_beam", "build_t_beam",
    "build_rectangular_composite_beam", "build_t_composite_beam",
    "build_l_composite_beam",
    "make_point_cloud",
    # 批量生成
    "DatasetGenerator",
]
