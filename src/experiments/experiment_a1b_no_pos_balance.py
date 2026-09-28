"""
消融 A1b — 无正样本平衡 (pos_ratio=1.0, 自然分布).

验证 pos_ratio=0.8 正样本平衡策略的必要性.
只训练 S1，仅改动 create_dataloaders(pos_ratio=1.0).
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._s1_trainer import train_s1
from lib.utils import console, CHECKPOINT_BASE

TAG = "ablation_A1b_pointmlp"
DESC = "A1b — 无正样本平衡 (pos_ratio=1.0)"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, TAG)


def main():
    train_s1(
        arch="pointmlp",
        num_epochs=80,
        tag=TAG,
        desc=DESC,
        ckpt_dir=CHECKPOINT_DIR,
        dataloader_kwargs={"pos_ratio": 1.0},
        val_batch_mult=4,
    )


if __name__ == "__main__":
    main()
