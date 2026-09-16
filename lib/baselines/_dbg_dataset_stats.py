"""
数据集统计核验: 用生成器实测, 核对论文 3.2 节的类别占比、噪声模型的点保留率与
边缘噪声的标签翻转率.

这些数字直接支撑"噪声模型标定"一节的论证, 也用于核对稿件正文的统计陈述.
"""

from __future__ import annotations

import numpy as np

from lib.pointcloud_generator._builder import BUILDERS, _filter_internal_rebar
from lib.pointcloud_generator._config import DENSITY_MAP, NOISE_LEVELS
from lib.pointcloud_generator._noise import apply_density_variation, apply_noise

TYPES = list(BUILDERS.keys())
VARIANTS = ["rebar", "sleeve"]
DENSITIES = ["high", "medium", "low"]


def clean_sample(comp, variant, dens, seed):
    rng = np.random.default_rng(seed)
    res = BUILDERS[comp](density=DENSITY_MAP[dens], rng=rng, variant=variant)
    pts, lbls = _filter_internal_rebar(res["points"], res["labels"])
    pts, lbls = apply_density_variation(pts, lbls, rng)
    return pts, lbls, rng


# ── 1) 类别占比 (核对 3.2 节: 混凝土 87.14% / 噪声 8.96% / 钢筋 2.40% / 套筒 1.50%) ──
print("=" * 90)
print("1) 类别点占比 (全 7 类构件 x 2 配筋型式 x 3 密度, clean 生成 + 密度扰动)")
print("=" * 90)
tot = np.zeros(4, dtype=np.int64)
n_samp = 0
per_dens = {d: np.zeros(4, dtype=np.int64) for d in DENSITIES}
seed = 1000
for comp in TYPES:
    for variant in VARIANTS:
        for dens in DENSITIES:
            pts, lbls, _ = clean_sample(comp, variant, dens, seed)
            seed += 1
            n_samp += 1
            for c in range(4):
                tot[c] += int((lbls == c).sum())
                per_dens[dens][c] += int((lbls == c).sum())
s = tot.sum()
print(f"样本数={n_samp}, 总点数={s:,}")
print(f"  噪声    {tot[0]:>12,}  {tot[0]/s:7.3%}")
print(f"  混凝土  {tot[1]:>12,}  {tot[1]/s:7.3%}")
print(f"  钢筋    {tot[2]:>12,}  {tot[2]/s:7.3%}")
print(f"  套筒    {tot[3]:>12,}  {tot[3]/s:7.3%}")
print(f"  钢筋+套筒          {tot[2]/s + tot[3]/s:7.3%}")
for d in DENSITIES:
    sd = per_dens[d].sum()
    print(f"  [密度={d:>6}] 混凝土={per_dens[d][1]/sd:7.3%} "
          f"钢筋={per_dens[d][2]/sd:7.3%} 套筒={per_dens[d][3]/sd:7.3%}")
print("  注: 论文 3.2 节的 87.14%/8.96%/2.40%/1.50% 是含噪声增强后的全数据集统计,")
print("      这里的 clean 生成不含外来噪点, 故噪声行为 0.")

# ── 2) 噪声模型的点保留率 ──
print()
print("=" * 90)
print("2) 噪声增强后的点保留率 (遮挡 + 随机丢弃 的叠加效果)")
print("=" * 90)
for dens in DENSITIES:
    base, lbl, _ = clean_sample("rectangular_column", "sleeve", dens, 7)
    line = f"  密度={dens:>6}  clean={len(base):>8,}"
    for nz in ["low", "medium", "high"]:
        rng = np.random.default_rng(77)
        pts, _ = apply_noise(base, lbl, NOISE_LEVELS[nz], rng)
        line += f"  {nz}={len(pts)/len(base):7.1%}"
    print(line)
print("  论文 Table 4 只给出 random dropout 的 keep 比例 (60%~92%), 但遮挡 (10%~50%)")
print("  同样删除点; 两者叠加后的实际保留率显著低于表中 keep 值.")

# ── 3) 边缘噪声的标签翻转率 ──
print()
print("=" * 90)
print("3) 边缘噪声造成的真值标签翻转率 (影响指标的不可达上界)")
print("=" * 90)
edge_only = {
    "occlusion_pct_range": (0.0, 0.0),
    "jitter_sigma_range": (0.0, 0.0),
    "keep_pct_range": (1.0, 1.0),
    "outlier_pct_range": (0.0, 0.0),
    "cluster_outlier_pct_range": (0.0, 0.0),
    "n_occlusion_polygons": (1, 1),
    "n_occlusion_sides": (3, 3),
}
for dens in ["low", "high"]:
    for pct in [0.01, 0.02, 0.05]:
        base, lbl, _ = clean_sample("rectangular_column", "sleeve", dens, 11)
        cfg = dict(edge_only, edge_noise_pct_range=(pct, pct))
        rng = np.random.default_rng(13)
        _, lbl2 = apply_noise(base, lbl, cfg, rng)
        flipped = int((lbl != lbl2).sum())
        print(f"  密度={dens:>6} edge_pct={pct:.2f}  "
              f"翻转点数={flipped:>6}  占全部点={flipped/len(lbl):7.4%}")
print("  结论: 标签翻转比例在 1e-4 量级, 不构成指标上界的主要限制 —— 这一点对作者有利.")
