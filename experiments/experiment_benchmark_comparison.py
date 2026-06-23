"""
基准模型对比试验: PointNet vs PointNet++ vs PointNeXt.

三个模型共享完全相同的数据流水线和损失函数,
仅更换骨干网络以量化架构差异对分割性能的影响.

运行方式:
  python experiments/experiment_benchmark_comparison.py

结果保存至:
  checkpoints/baseline_pointnet/summary_baseline_pointnet.json
  checkpoints/baseline_pointnet2/summary_baseline_pointnet2.json
  checkpoints/baseline_pointnext/summary_baseline_pointnext.json
  checkpoints/benchmark_comparison_summary.json
"""

import json
import os
from rich.table import Table

from _common import (
    console,
    CHECKPOINT_BASE,
    run_experiment,
    PointNetSeg,
    PointNetPlusPlus,
    PointNeXtSeg,
)

BENCHMARK_CONFIGS = [
    {
        "model_factory": lambda: PointNetSeg(num_classes=3),
        "tag": "baseline_pointnet",
        "desc": "基准: PointNet",
    },
    {
        "model_factory": lambda: PointNetPlusPlus(num_classes=3),
        "tag": "baseline_pointnet2",
        "desc": "主试验: PointNet++",
    },
    {
        "model_factory": lambda: PointNeXtSeg(num_classes=3),
        "tag": "baseline_pointnext",
        "desc": "基准: PointNeXt",
    },
]


def print_comparison(results: list):
    """打印基准模型对比表."""
    table = Table(title="基准模型对比", title_style="bold cyan")
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
    console.rule("[bold cyan]基准模型对比试验[/]")
    console.print("共享相同数据流水线 + PerSampleWeightedCE + 噪声感知\n")

    results = []
    for cfg in BENCHMARK_CONFIGS:
        console.print(f"\n[bold]══════▶ {cfg['desc']}[/]")
        summary = run_experiment(**cfg)
        results.append(summary)

    # 汇总与保存
    console.rule("[bold cyan]基准模型对比汇总[/]")
    print_comparison(results)

    summary_path = os.path.join(CHECKPOINT_BASE, "benchmark_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold green]保存至: {summary_path}[/]")


if __name__ == "__main__":
    main()
