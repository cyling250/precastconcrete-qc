"""
消融 A1a — 无局部块采样 (纯随机采样).

验证正样本驱动的局部块采样的必要性.
只训练 S1，仅改动 create_dataloaders(random_sample=True).
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._s1_trainer import train_s1
from lib.utils import console, CHECKPOINT_BASE

TAG = "ablation_A1a_pointmlp"
DESC = "A1a — 无局部块采样 (纯随机采样)"
CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, TAG)


def main():
    train_s1(
        arch="pointmlp",
        num_epochs=120,
        tag=TAG,
        desc=DESC,
        ckpt_dir=CHECKPOINT_DIR,
        dataloader_kwargs={"random_sample": True},
    )


if __name__ == "__main__":
    main()
