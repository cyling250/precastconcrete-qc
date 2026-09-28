"""
消融 A4 — 单阶段 4 类 + Focal Loss.

补充类不平衡替代方案对照: 验证 focal loss 能否替代级联设计.
回应 Limitations 中 "does not include focal loss" 的表述.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._a2_trainer import train_a2

TAG = "ablation_A4_focal"
DESC = "A4 — 单阶段 4 类 + Focal Loss (γ=2, 无 α)"


def main():
    train_a2(tag=TAG, desc=DESC, loss_mode="focal", val_batch_mult=4)


if __name__ == "__main__":
    main()
