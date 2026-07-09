"""
合成数据集分析: 类别分布、构件分布、密度/噪声覆盖、点数统计、可视化.
"""

import sys, os, json, glob, re, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

DATA_DIRS = {
    "train": os.path.join("mock", "inuse", "train"),
    "val": os.path.join("mock", "inuse", "val"),
    "test": os.path.join("mock", "inuse", "test"),
}

COMPONENT_NAMES = {
    "rectangular_column": "矩形柱",
    "l_shaped_column": "L形柱",
    "rectangular_beam": "矩形梁",
    "t_beam": "T形梁",
    "rectangular_composite_beam": "矩形叠合梁",
    "t_composite_beam": "T形叠合梁",
    "l_composite_beam": "L形叠合梁",
}

LABEL_NAMES = {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"}
LABEL_COLORS = {0: "blue", 1: "grey", 2: "red", 3: "yellow"}

TOTAL_EXPECTED = (
    7 * 10 * 2 * 3 * 3
)  # 7 types × 10 ids × 2 variants × 3 noise × 3 density

# ── 文件解析 ──

FILENAME_RE = re.compile(
    r"Mock_(?P<type>\w+)_(?P<id>\d+)_(?P<variant>rebar|sleeve)_"
    r"(?P<noise>low|medium|high)_(?P<density>low|medium|high)\.npz"
)


def parse_filename(fname: str) -> dict | None:
    m = FILENAME_RE.match(fname)
    if m is None:
        return None
    return m.groupdict()


# ═══════════════════════════════════════════════════════════
#  1. 类别分布
# ═══════════════════════════════════════════════════════════


def compute_class_distribution():
    """统计所有 split 中各类点数占比."""
    console.rule("[bold]1. 类别分布统计[/]")

    counts = Counter()
    total_files = 0
    for split, d in DATA_DIRS.items():
        for fpath in sorted(glob.glob(os.path.join(d, "*.npz"))):
            data = np.load(fpath)
            lbls = data["labels"]
            counts.update(lbls.tolist())
            total_files += 1

    total_pts = sum(counts.values())
    table = Table(box=box.ROUNDED)
    table.add_column("类别")
    table.add_column("点数")
    table.add_column("占比")

    for c in sorted(counts):
        name = LABEL_NAMES.get(c, f"类{c}")
        pct = counts[c] / total_pts * 100
        table.add_row(name, f"{counts[c]:,}", f"{pct:.2f}%")

    console.print(table)
    console.print(f"[dim]统计: {total_files} 文件, {total_pts:,} 总点数[/]")

    return {
        "total_files": total_files,
        "total_points": total_pts,
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "class_pct": {int(k): float(v / total_pts * 100) for k, v in counts.items()},
    }


# ═══════════════════════════════════════════════════════════
#  2. 构件分布
# ═══════════════════════════════════════════════════════════


def compute_component_distribution():
    """按构件类型和配筋方案统计."""
    console.rule("[bold]2. 构件分布统计[/]")

    type_counts = Counter()
    variant_counts = Counter()
    total_files = 0

    for split, d in DATA_DIRS.items():
        for fpath in sorted(glob.glob(os.path.join(d, "*.npz"))):
            info = parse_filename(os.path.basename(fpath))
            if info is None:
                continue
            type_counts[info["type"]] += 1
            variant_counts[info["variant"]] += 1
            total_files += 1

    table = Table(box=box.ROUNDED)
    table.add_column("构件类型")
    table.add_column("文件数")
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        name = COMPONENT_NAMES.get(t, t)
        table.add_row(name, str(cnt))
    table.add_row("[bold]合计[/]", f"[bold]{total_files}[/]")
    console.print(table)

    vtable = Table(box=box.ROUNDED)
    vtable.add_column("配筋方案")
    vtable.add_column("文件数")
    vtable.add_column("占比")
    for v, cnt in sorted(variant_counts.items()):
        vtable.add_row(v, str(cnt), f"{cnt / total_files * 100:.1f}%")
    vtable.add_row("[bold]合计[/]", f"[bold]{total_files}[/]", "[bold]100%[/]")
    console.print(vtable)

    return {
        "total_files": total_files,
        "type_counts": dict(type_counts),
        "variant_counts": dict(variant_counts),
    }


# ═══════════════════════════════════════════════════════════
#  3. 密度/噪声覆盖
# ═══════════════════════════════════════════════════════════


def compute_coverage():
    """各密度×噪声等级组合的文件数."""
    console.rule("[bold]3. 密度×噪声等级覆盖[/]")

    combos = Counter()
    for split, d in DATA_DIRS.items():
        for fpath in sorted(glob.glob(os.path.join(d, "*.npz"))):
            info = parse_filename(os.path.basename(fpath))
            if info is None:
                continue
            combos[(info["density"], info["noise"])] += 1

    levels = ["low", "medium", "high"]
    table = Table(box=box.ROUNDED)
    table.add_column("密度\\噪声")
    for n in levels:
        table.add_column(n)
    for d in levels:
        row = [d]
        for n in levels:
            row.append(str(combos.get((d, n), 0)))
        table.add_row(*row)
    console.print(table)
    console.print(f"[dim]总计 {sum(combos.values())} 文件[/]")

    return {"combos": {f"{d}_{n}": combos[(d, n)] for d in levels for n in levels}}


# ═══════════════════════════════════════════════════════════
#  4. 点数统计
# ═══════════════════════════════════════════════════════════


def compute_point_stats():
    """每样本点数分布."""
    console.rule("[bold]4. 每样本点数统计[/]")

    n_pts_list = []
    for split, d in DATA_DIRS.items():
        for fpath in sorted(glob.glob(os.path.join(d, "*.npz"))):
            data = np.load(fpath)
            n_pts_list.append(len(data["points"]))

    n_pts = np.array(n_pts_list)
    table = Table(box=box.ROUNDED)
    table.add_column("统计量")
    table.add_column("值")
    for label, val in [
        ("总数", str(len(n_pts))),
        ("平均值", f"{n_pts.mean():.0f}"),
        ("标准差", f"{n_pts.std():.0f}"),
        ("最小值", str(n_pts.min())),
        ("P25", f"{np.percentile(n_pts, 25):.0f}"),
        ("中位数", f"{np.percentile(n_pts, 50):.0f}"),
        ("P75", f"{np.percentile(n_pts, 75):.0f}"),
        ("最大值", str(n_pts.max())),
    ]:
        table.add_row(label, val)
    console.print(table)

    return {
        "count": len(n_pts),
        "mean": float(n_pts.mean()),
        "std": float(n_pts.std()),
        "min": int(n_pts.min()),
        "p25": float(np.percentile(n_pts, 25)),
        "p50": float(np.percentile(n_pts, 50)),
        "p75": float(np.percentile(n_pts, 75)),
        "max": int(n_pts.max()),
    }


# ═══════════════════════════════════════════════════════════
#  5. 可视化样本选择
# ═══════════════════════════════════════════════════════════


def pick_visualization_samples():
    """为7种构件各选一个代表性样本 (排除噪声后点数最多的)."""
    console.rule("[bold]5. 可视化样本选择[/]")

    best = {}
    for fpath in sorted(glob.glob(os.path.join("mock", "inuse", "train", "*.npz"))):
        info = parse_filename(os.path.basename(fpath))
        if info is None:
            continue
        t = info["type"]
        if t not in best:
            data = np.load(fpath)
            lbls = data["labels"]
            n_non_noise = (lbls != 0).sum()
            best[t] = (fpath, n_non_noise, info)

    console.print(f"已为 {len(best)} 种构件选择可视化样本:")
    for t in sorted(best, key=lambda k: COMPONENT_NAMES.get(k, k)):
        fpath, n, info = best[t]
        name = COMPONENT_NAMES.get(t, t)
        console.print(f"  {name}: {os.path.basename(fpath)} ({n:,} 非噪声点)")
    return best


def export_visualization(best: dict, out_dir: str = "docs/figs/dataset"):
    """
    用 Open3D 渲染 7 个样本（按标签着色）, 保存相机姿态文件供手动截图.

    运行后会在 out_dir 生成:
      - {type}_pointcloud.npy: (N, 3) 点坐标
      - {type}_labels.npy:     (N,) 标签
      - {type}_camera.json:   初始相机参数 (可手动调整后保存)

    用户运行 `python experiments/experiment_dataset_analysis.py --visualize`
    后用 Open3D 可视化界面调整角度, 按 'c' 截取当前视角截图保存.
    """
    import open3d as o3d

    os.makedirs(out_dir, exist_ok=True)

    LABEL_RGB = {
        0: (0.3, 0.5, 0.9),  # 噪声-蓝
        1: (0.6, 0.6, 0.6),  # 混凝土-灰
        2: (0.9, 0.2, 0.2),  # 钢筋-红
        3: (0.9, 0.9, 0.2),  # 套筒-黄
    }

    # 启用命令行交互
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

    items = sorted(best.items(), key=lambda x: COMPONENT_NAMES.get(x[0], x[0]))

    for i, (t, (fpath, n_pts, info)) in enumerate(items):
        data = np.load(fpath)
        pts = data["points"]
        lbls = data["labels"]

        center = pts.mean(axis=0)
        pts_unit = pts - center
        s = float(np.linalg.norm(np.ptp(pts_unit, axis=0)))
        pts_unit /= max(s, 1e-8)

        colors = np.array([LABEL_RGB.get(l, (0, 0, 0)) for l in lbls])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_unit)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        np.save(os.path.join(out_dir, f"{t}_pointcloud.npy"), pts)
        np.save(os.path.join(out_dir, f"{t}_labels.npy"), lbls)

        name = COMPONENT_NAMES.get(t, t)
        console.print(f"\n[bold cyan]▶ {name}[/] ({i + 1}/{len(items)})")
        console.print("  按 [bold]'c'[/] 截图, 关闭窗口进入下一个")

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=720)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.run()
        vis.destroy_window()

    console.print(f"\n[bold green]可视化完成![/]")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    console.print("[bold]合成数据集分析[/]\n")

    cls_result = compute_class_distribution()
    comp_result = compute_component_distribution()
    cov_result = compute_coverage()
    pts_result = compute_point_stats()

    results = {
        "class_distribution": cls_result,
        "component_distribution": comp_result,
        "coverage": cov_result,
        "point_stats": pts_result,
    }

    with open("experiments/checkpoints/dataset_analysis.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(
        f"[green]分析结果已保存: experiments/checkpoints/dataset_analysis.json[/]"
    )

    # 可视化样本选择 (仅打印)
    best = pick_visualization_samples()

    console.print("\n[bold]若要启动可视化:[/]")
    console.print("  python experiments/experiment_dataset_analysis.py --visualize")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="启动 Open3D 可视化")
    args = parser.parse_args()

    if args.visualize:
        best = pick_visualization_samples()
        export_visualization(best)
    else:
        main()
