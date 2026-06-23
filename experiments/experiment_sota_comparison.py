"""
SOTA 模型对比试验: PointNeXt vs PointMLP (非 Transformer SOTA 模型).

两个模型共享完全相同的数据流水线和损失函数,
验证本文方法 + 不同 SOTA 骨干网络的组合性能.

选型说明 (基于 2022-2025 点云语义分割文献调研):
  - PointNeXt (Qian et al., NeurIPS 2022): 在 S3DIS 上 mIoU=74.9%,
    超越 Point Transformer, 推理速度是 PointMLP 的 10×.
  - PointMLP (Ma et al., ICLR 2022): 纯 MLP 架构, 在 ScanObjectNN 上
    OA=86.2%, 无需复杂算子.
  两者均为非 Transformer 架构, 参数量适中可训练.

运行方式:
  python experiments/experiment_sota_comparison.py

结果保存至:
  checkpoints/baseline_pointnext/summary_baseline_pointnext.json
  checkpoints/baseline_pointmlp/summary_baseline_pointmlp.json
  checkpoints/sota_comparison_summary.json
"""

import json
import os
from rich.table import Table

from _common import (
    console,
    CHECKPOINT_BASE,
    run_experiment,
    PointNeXtSeg,
    PointMLPSeg,
)

SOTA_CONFIGS = [
    {
        "model_factory": lambda: PointNeXtSeg(num_classes=3),
        "tag": "baseline_pointnext",
        "desc": "SOTA: PointNeXt (NeurIPS 2022)",
    },
    {
        "model_factory": lambda: PointMLPSeg(num_classes=3),
        "tag": "baseline_pointmlp",
        "desc": "SOTA: PointMLP (ICLR 2022)",
    },
]


def print_comparison(results: list):
    """打印 SOTA 模型对比表."""
    table = Table(title="SOTA 模型对比", title_style="bold cyan")
    table.add_column("模型", style="cyan", no_wrap=True)
    table.add_column("val_mIoU", justify="right")
    table.add_column("噪声IoU", justify="right")
    table.add_column("混凝土IoU", justify="right")
    table.add_column("钢筋IoU", justify="right")
    table.add_column("干净mIoU", justify="right")
    table.add_column("test_mIoU", justify="right")
    table.add_column("test_acc", justify="right")

    for r in results:
        table.add_row(
            r.get("tag", "?"),
            f"{r.get('best_val_miou', 0):.4f}",
            f"{r.get('best_val_noise', 0):.4f}",
            f"{r.get('best_val_concrete', 0):.4f}",
            f"{r.get('best_val_rebar', 0):.4f}",
            f"{r.get('best_val_clean_miou', 0):.4f}",
            f"{r.get('test_mIoU', 0):.4f}",
            f"{r.get('test_acc', 0):.4f}",
        )

    console.print(table)


def main():
    console.rule("[bold cyan]SOTA 模型对比试验[/]")
    console.print("PointNeXt vs PointMLP — 共享相同数据流水线 + loss\n")

    results = []
    for cfg in SOTA_CONFIGS:
        console.print(f"\n[bold]══════▶ {cfg['desc']}[/]")
        summary = run_experiment(**cfg)
        results.append(summary)

    # 汇总与保存
    console.rule("[bold cyan]SOTA 模型对比汇总[/]")
    print_comparison(results)

    summary_path = os.path.join(CHECKPOINT_BASE, "sota_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold green]保存至: {summary_path}[/]")


if __name__ == "__main__":
    main()
