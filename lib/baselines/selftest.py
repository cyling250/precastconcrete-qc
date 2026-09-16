"""
几何基线自检: 用项目自身的点云生成器造构件, 在已知真值上验证几何分割逻辑.

价值: 不需要 CUDA、不需要 mock 数据集, 纯 CPU 就能确认几何基线跑得通、
以及它在低/中/高噪声下的表现量级, 从而支撑"噪声-退化交叉点"分析.

用法:
    python -m lib.baselines.selftest
    python -m lib.baselines.selftest --types rectangular_column --noise low
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lib.pointcloud_generator._builder import BUILDERS, _filter_internal_rebar
from lib.pointcloud_generator._config import DENSITY_MAP, NOISE_LEVELS
from lib.pointcloud_generator._noise import apply_density_variation, apply_noise

from .geometric_seg import (
    GeometricParams,
    compute_iou_np,
    rs_iou_np,
    segment_geometric,
)

CLASS_NAMES = {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"}


def build_sample(comp_type: str, variant: str, density: str, noise: str, seed: int):
    """按生成器原始流程造一个带真值的构件 (与 mock 数据同分布)."""
    rng = np.random.default_rng(seed)
    builder = BUILDERS[comp_type]
    res = builder(density=DENSITY_MAP[density], rng=rng, variant=variant)
    pts, lbls = _filter_internal_rebar(res["points"], res["labels"])
    pts, lbls = apply_density_variation(pts, lbls, rng)
    if noise != "clean":
        pts, lbls = apply_noise(pts, lbls, NOISE_LEVELS[noise], rng)
    return pts, lbls


def main() -> None:
    ap = argparse.ArgumentParser(description="几何基线自检")
    ap.add_argument("--types", nargs="*", default=["rectangular_column", "t_composite_beam"])
    ap.add_argument("--variants", nargs="*", default=["rebar", "sleeve"])
    ap.add_argument("--densities", nargs="*", default=["low"])
    ap.add_argument("--noise", nargs="*", default=["clean", "low", "medium", "high"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--four-class", action="store_true",
                    help="跑四分类口径 (几何方法给不出套筒类, 仅供对照)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    three = not args.four_class
    names = CLASS_NAMES if three else CLASS_NAMES
    print("=" * 96)
    print(f"几何基线自检 (PCA 特征 + RANSAC 圆柱/平面, 纯 CPU) "
          f"{'三分类口径 噪声/混凝土/金属' if three else '四分类口径'}")
    print("=" * 96)

    for comp in args.types:
        for variant in args.variants:
            for dens in args.densities:
                for noise in args.noise:
                    t0 = time.time()
                    pts, gt = build_sample(comp, variant, dens, noise, args.seed)
                    if len(pts) < 1000:
                        continue
                    if three:
                        # 真值也合并为 噪声/混凝土/金属, 与 Stage 1 同口径
                        gt = gt.copy()
                        gt[gt == 3] = 2
                        names = {0: "噪声", 1: "混凝土", 2: "金属"}
                    pred, info = segment_geometric(
                        pts, GeometricParams(seed=args.seed, three_class=three),
                        return_info=True,
                    )
                    ious = compute_iou_np(pred, gt, names)
                    cm = float(np.mean([v for k, v in ious.items() if k != "mIoU"]))
                    rs = rs_iou_np(pred, gt)
                    io = " | ".join(
                        f"{k}={v:.3f}" for k, v in ious.items() if k != "mIoU"
                    )
                    print(
                        f"\n[{comp} / {variant} / dens={dens} / noise={noise}]  "
                        f"N={len(pts):,}  用时 {time.time() - t0:.1f}s"
                    )
                    print(f"  {io}")
                    print(
                        f"  mIoU={ious['mIoU']:.3f}  cm={cm:.3f}  rsIoU={rs:.3f}  "
                        f"套筒圆柱={info.get('n_cyl_sleeve', 0)}{info.get('cyl_radii', [])}  "
                        f"金属点={info.get('n_metal_pts', info.get('n_rebar_pts', 0))}  "
                        f"平面={info.get('n_planes', '-')}"
                    )
                    if not args.quiet:
                        gt_frac = {
                            int(c): round(float(np.mean(gt == c)), 4) for c in range(4)
                        }
                        print(f"  预测占比: {info['frac_full']}   真值占比: {gt_frac}")


if __name__ == "__main__":
    main()
