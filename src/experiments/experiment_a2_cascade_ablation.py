"""
消融 A2 — 无级联管线: 单阶段 4 类 PointMLP 直接分割.

验证级联自适应体素降采样管线的必要性.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._a2_trainer import train_a2
from lib.utils import console


def main():
    train_a2(val_batch_mult=4)


if __name__ == "__main__":
    main()
