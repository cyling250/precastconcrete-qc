"""
点云分割指标: IoU, Precision/Recall/F1, Accuracy.

复用函数, 可被 experiment_*.py 通用引用.
"""

import numpy as np
import torch


@torch.no_grad()
def compute_iou(pred: torch.Tensor, target: torch.Tensor, class_names: dict) -> dict:
    """返回 {class_name: IoU, 'mIoU': mean}.

    Args:
        pred: (B, N, C) logits 或 (B*N,) argmax 标签
        target: (B, N) 标签
        class_names: {0: "噪声", 1: "混凝土", 2: "钢筋"} 等
    """
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred = pred.argmax(dim=-1)
    pred = pred.flatten()
    target = target.flatten()
    ious = {}
    for c, name in class_names.items():
        inter = ((pred == c) & (target == c)).sum().item()
        union = ((pred == c) | (target == c)).sum().item()
        ious[name] = inter / (union + 1e-6)
    vals = [v for k, v in ious.items() if k != "mIoU"]
    ious["mIoU"] = float(np.mean(vals)) if vals else 0.0
    return ious


@torch.no_grad()
def compute_precision_recall_f1(
    pred: torch.Tensor, target: torch.Tensor, class_names: dict
) -> dict:
    """返回 {class_name: {precision, recall, f1}}."""
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred = pred.argmax(dim=-1)
    pred = pred.flatten()
    target = target.flatten()
    metrics = {}
    for c, name in class_names.items():
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


@torch.no_grad()
def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred = pred.argmax(dim=-1)
    return (pred == target).float().mean().item()
