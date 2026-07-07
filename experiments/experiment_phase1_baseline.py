"""
Phase 1 Baseline: S1 三分类 (噪声/混凝土/金属).

使用方法:
  python experiment_phase1_baseline.py                              # PointNet++ (默认)
  python experiment_phase1_baseline.py --arch pointnet               # PointNet
  python experiment_phase1_baseline.py --arch pointmlp               # PointMLP
  python experiment_phase1_baseline.py --arch pointnext              # PointNeXt
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from experiments._config import ARCH_REGISTRY, S1_DEFAULT_EPOCHS, S1_DEFAULT_PATIENCE
from experiments._s1_trainer import train_s1
from lib.utils import console


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: S1 三分类基线")
    parser.add_argument(
        "--arch",
        type=str,
        default="pointnet++",
        choices=ARCH_REGISTRY,
        help="骨干网络架构 (默认: pointnet++)",
    )
    parser.add_argument("--run-suffix", type=str, default="", help="自定义标签后缀")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--patience", type=int, default=None, help="早停耐心值")
    return parser.parse_args()


def main():
    args = parse_args()
    arch = args.arch
    run_suffix = args.run_suffix
    tag = f"phase1_baseline_{arch}{run_suffix}"
    desc = f"Phase 1: S1 ({arch}, 噪声/混凝土/金属)"

    num_epochs = args.epochs if args.epochs is not None else S1_DEFAULT_EPOCHS
    early_stop_patience = (
        args.patience if args.patience is not None else S1_DEFAULT_PATIENCE
    )

    train_s1(
        arch=arch,
        num_epochs=num_epochs,
        early_stop_patience=early_stop_patience,
        tag=tag,
        desc=desc,
    )


if __name__ == "__main__":
    main()
