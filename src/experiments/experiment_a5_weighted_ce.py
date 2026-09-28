"""
消融 A5 — 单阶段 4 类 + 类加权交叉熵.

补充类不平衡替代方案对照: 验证逐样本类加权 CE 能否替代级联设计.
回应 Limitations 中 "does not include class-weighted cross-entropy" 的表述.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._a2_trainer import train_a2

TAG = "ablation_A5_wce"
DESC = "A5 — 单阶段 4 类 + 类加权交叉熵 (w=N/(4·n_c), 噪声×0.5)"


def main():
    train_a2(tag=TAG, desc=DESC, loss_mode="wce", val_batch_mult=4)


if __name__ == "__main__":
    main()
