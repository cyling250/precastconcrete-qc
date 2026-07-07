"""
Phase 2 Baseline: S2 四类精分割 (噪声/混凝土/钢筋/套筒).

管线: S1 → 混凝土4cm体素降采样 → S2

使用方法:
  python experiment_phase2_baseline.py                              # PointNet++ (默认)
  python experiment_phase2_baseline.py --arch pointnet               # S1=PointNet
  python experiment_phase2_baseline.py --arch pointmlp               # S1=PointMLP
  python experiment_phase2_baseline.py --arch pointnext              # S1=PointNeXt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from experiments._config import ARCH_REGISTRY
from experiments._s2_trainer import train_s2


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: S2 四类精分割基线")
    parser.add_argument("--arch", type=str, default="pointnet++",
                        choices=ARCH_REGISTRY, help="S1 骨干网络架构 (默认: pointnet++)")
    return parser.parse_args()


def main():
    args = parse_args()
    train_s2(arch=args.arch)


if __name__ == "__main__":
    main()
