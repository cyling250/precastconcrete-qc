# PC-CascadeNet: Point-Cloud-Based Material-Level Segmentation of Precast Concrete Components

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%20CUDA%2013.0-red)](https://pytorch.org/)

Companion repository for the manuscript *"Point-cloud-based material level segmentation of precast concrete components for automated quality inspection using a two-stage cascade network"* (submitted to the *Journal of Building Engineering*).

**Languages / 语言：[English](README.md) | [中文](README.zh-CN.md)**

## Overview

Dimensional quality inspection of precast concrete (PC) components is the gateway to quality control in prefabricated construction, and laser scanning has become the standard way to perform it. Turning a scan into an inspection result requires segmenting the point cloud into four material-level classes — **concrete, rebar, grouting sleeve, and noise** — which is hampered by severe on-site scanning noise, an extreme class imbalance (rebar and sleeve points are scarce but structurally critical), and the scarcity of annotated real-world data.

This repository provides three things:

1. **PC-CascadeNet**, a two-stage cascade segmentation network. Stage 1 performs coarse three-class segmentation (noise / concrete / rebar-sleeve union); after adaptive voxel downsampling of the concrete points, stage 2 conducts fine four-class segmentation. A local patch sampling strategy with class-balanced sampling (150 mm ball queries, matching the typical rebar-spacing range of GB 50010-2010) secures sufficient training signal for the sparse classes. The core insight: the dominant, geometrically simple concrete class interferes with learning the scarce rebar and sleeve, so the cascade lets each stage solve its own sub-problem.
2. **A parametric synthetic data generator** grounded in the Chinese engineering specifications JGJ 355-2015 (grout sleeve splicing) and GB 50010-2010 (concrete structure design). It synthesizes **1,260 labeled point clouds**: 7 PC component types × 2 reinforcement variants × 3 density levels × 3 coupled noise levels (six coupled noise types replicating on-site scanning), with consistent point density across materials.
3. **A complete experiment suite**: architecture comparisons (PointMLP, PointNet, PointNet++, PointNeXt in both cascade stages), ablations (random sampling / no positive-balance / single-stage / focal loss / weighted CE), a calibrated classical geometric baseline (PCA descriptors + RANSAC), sensitivity and robustness analyses (patch radius, voxel size, density, noise), and a **segmentation → measurement closed loop** that checks rebar spacing and sleeve offsets against code tolerances.

## Headline results

- **91.8% mean IoU** on the synthetic test set, outperforming the three mainstream architectures in both cascade stages.
- In downstream measurement, PC-CascadeNet is **the only tested variant that keeps longitudinal-bar spacing errors within the ±10 mm installation tolerance up to medium noise levels** — the segmentation gain carries through to inspection practice.
- Against the calibrated geometric baseline in the measurement loop: sleeve matching 14/16 components vs 0/16, and longitudinal-bar spacing error p90 = 1.86 mm (84% of components valid) vs 96 mm (45% valid).

## Repository layout

```
precastconcrete-qc/
├── src/
│   ├── lib/
│   │   ├── pointcloud_generator/   # parametric dataset generator (7 component types)
│   │   ├── cascade/                # two-stage cascade models (S1/S2)
│   │   ├── backbones/              # PointMLP / PointNet / PointNet++ / PointNeXt
│   │   ├── baselines/              # geometric baseline (PCA + RANSAC) + diagnostics
│   │   ├── measurement/            # segmentation → dimensional measurement loop
│   │   └── utils/                  # shared constants, datasets, training, metrics
│   └── experiments/                # experiment entry scripts (see table below)
├── scripts/                        # queue runner, plotting, result cross-checks
├── docs/                           # development handover document (Chinese)
├── mock/                           # generated dataset (gitignored, ~6.9 GB local)
└── out/                            # weights, results, logs (gitignored, local only)
```

## Experiment naming

| Prefix | Family | Scripts |
|---|---|---|
| `phase1` / `phase2` | Main architecture baselines (stage 1 / stage 2) | `experiment_phase1_baseline.py`, `experiment_phase2_baseline.py` |
| `a1a`–`a5` | Ablations (paper models M1–M5) | random sampling, no positive-balance, single-stage (M3), focal loss, weighted CE |
| `b1`, `b2` | Classical geometric baseline | full sweep (`out/B1`), val-calibrated test report (`out/E4`) |
| `e1`, `e3`, `e5` | Mechanism & closed loop | unified-protocol tables, FPS survival rates, segmentation→measurement loop |
| `s1`–`s4` | Sensitivity / robustness | patch radius, voxel size, density, noise (11-level axis; `--all-keys` = legacy leakage control) |

Files prefixed with `_` in `src/experiments/` are helper modules (trainers, shared config, full-cloud prediction), not entry points.

## Quick start

Requirements: Python 3.10, an NVIDIA GPU (reference runs used a RTX 4060 Ti 16 GB).

```bash
# 1. Install dependencies (repo root; torch comes from the cu130 index)
uv sync

# 2. Generate the full 1,260-sample dataset into mock/inuse/
cd src
../.venv/Scripts/python.exe -m lib.pointcloud_generator --batch

# 3. Train the two-stage baselines (repeat with --arch pointnet / pointnet++ / pointnext)
../.venv/Scripts/python.exe -m experiments.experiment_phase1_baseline --arch pointmlp
../.venv/Scripts/python.exe -m experiments.experiment_phase2_baseline --arch pointmlp

# 4. Evaluate / reproduce paper numbers (run from src/, results land in out/)
../.venv/Scripts/python.exe -m experiments.experiment_e1_tables --out out/E1        # comparison tables
../.venv/Scripts/python.exe -m experiments.experiment_s4_noise                      # noise robustness
../.venv/Scripts/python.exe -m experiments.experiment_e5_measurement --out out/E5 --deep   # measurement loop
```

The ablation, geometric-baseline, and remaining sensitivity scripts follow the same pattern (`python -m experiments.experiment_<id>_<name>`). `scripts/queue_runner.py` runs the full supplementary-experiment queue idempotently and can resume from checkpoints. Experiment outputs are written under `out/` and are not committed; the dataset under `mock/` is regenerable with the seeded generator.

## Data availability

The dataset and source code supporting the manuscript are openly available in this repository. Since the labeled dataset is produced by the deterministic parametric generator, it can be reproduced locally with the command above instead of being stored in git.

## Citation

If this repository helps your research, please cite the manuscript (citation to be added upon publication).
