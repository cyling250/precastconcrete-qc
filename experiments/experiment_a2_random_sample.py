"""
消融 A2 — 删除物理感知局部块采样: 使用纯随机点云采样.

与主试验的唯一区别: create_dataloaders(random_sample=True).
"""
import torch.nn as nn
from _common import run_experiment, PointNetPlusPlus, NUM_CLASSES


def main():
    run_experiment(
        model_factory=lambda: PointNetPlusPlus(num_classes=NUM_CLASSES, in_channels=3),
        tag="ablation_A2",
        desc="消融 A2 — 删除物理感知采样 (纯随机采样)",
        criterion=nn.CrossEntropyLoss(),
        random_sample=True,
    )


if __name__ == "__main__":
    main()
