# -*- coding: utf-8 -*-
"""
@File    :   preview_mock.py
@Time    :   2026/06/18
@Desc    :   Mock 点云预览脚本 - 可视化合成数据集

用法：
  python preview_mock.py                                    # 预览第一个找到的 .ply 文件
  python preview_mock.py --file Mock_rectangular_column_001_high.ply  # 预览指定文件
  python preview_mock.py --list                             # 列出所有可用文件
  python preview_mock.py --all                              # 并列预览所有文件
"""

import os
import sys
import argparse
import glob

import numpy as np
import open3d as o3d

POINT_DIR = os.path.join(os.path.dirname(__file__), "res", "point")


def list_files() -> list:
    """列出所有生成的 Mock 点云文件。"""
    if not os.path.isdir(POINT_DIR):
        return []
    files = sorted(glob.glob(os.path.join(POINT_DIR, "Mock_*.ply")))
    return files


def print_file_list(files: list):
    """打印文件列表（含大小和点数）。"""
    if not files:
        print("未找到 Mock 点云文件，请先运行 generate_synthetic_dataset.py")
        return

    print(f"{'文件名':<55} {'大小':>10} {'点数':>10}")
    print("-" * 77)
    for f in files:
        size_kb = os.path.getsize(f) / 1024
        try:
            pcd = o3d.io.read_point_cloud(f)
            n_pts = len(pcd.points)
        except Exception:
            n_pts = 0
        print(f"{os.path.basename(f):<55} {size_kb:>8.1f}KB {n_pts:>10}")


def preview_file(filepath: str, show_axes: bool = True):
    """预览单个点云文件。

    Args:
        filepath: .ply 文件路径
        show_axes: 是否显示坐标轴
    """
    if not os.path.exists(filepath):
        # 尝试在默认目录查找
        alt = os.path.join(POINT_DIR, os.path.basename(filepath))
        if os.path.exists(alt):
            filepath = alt
        else:
            print(f"文件不存在: {filepath}")
            return

    print(f"加载: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    n_pts = len(pcd.points)
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()

    print(f"  点数: {n_pts}")
    print(f"  颜色: {'有' if has_colors else '无'}")
    print(f"  法向: {'有' if has_normals else '无'}")

    # 边界框
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"  范围: [{bbox.min_bound}] → [{bbox.max_bound}]")
    dims = bbox.max_bound - bbox.min_bound
    print(f"  尺寸: {dims[0]:.0f} × {dims[1]:.0f} × {dims[2]:.0f} mm")

    # 分析颜色分布（区分混凝土/钢筋）
    if has_colors:
        colors = np.asarray(pcd.colors)
        # 暗红色钢筋 (R>0.5, G<0.4, B<0.4)
        rebar_mask = (colors[:, 0] > 0.5) & (colors[:, 1] < 0.4) & (colors[:, 2] < 0.4)
        n_rebar = rebar_mask.sum()
        n_concrete = n_pts - n_rebar
        print(f"  混凝土点: {n_concrete} ({100 * n_concrete / n_pts:.1f}%)")
        print(f"  钢筋点:   {n_rebar} ({100 * n_rebar / n_pts:.1f}%)")

    # 可视化
    geometries = [pcd]
    if show_axes:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(dims) * 0.15, origin=bbox.min_bound
        )
        geometries.append(axis)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=os.path.basename(filepath),
        width=1024,
        height=768,
    )


def preview_all():
    """依次预览所有 Mock 点云文件（每种类型选一个 high 样本）。"""
    files = list_files()
    if not files:
        print("未找到 Mock 点云文件")
        return

    mock_files = [f for f in files if os.path.basename(f).startswith("Mock_")]
    if not mock_files:
        print("未找到 Mock_ 前缀的文件")
        return

    # 按类型分组预览（每种类型选一个 high 密度）
    types_seen = set()
    selected = []
    for f in mock_files:
        name = os.path.basename(f)
        parts = name.replace("Mock_", "").split("_")
        comp_type = "_".join(parts[:-2])
        if comp_type not in types_seen and "_high" in name:
            types_seen.add(comp_type)
            selected.append(f)

    if not selected:
        selected = [mock_files[0]]

    print(f"\n预览 {len(selected)} 种构件类型（各选一个 high 密度样本）:")
    for f in selected:
        print(f"  {os.path.basename(f)}")

    for f in selected:
        preview_file(f)


def browse_all():
    """交互式浏览所有 Mock 点云文件（← → 方向键切换，Esc 退出）。"""
    mock_files = sorted(
        f for f in list_files() if os.path.basename(f).startswith("Mock_")
    )
    if not mock_files:
        print("未找到 Mock 点云文件")
        return

    # 预加载所有点云
    print(f"加载 {len(mock_files)} 个点云文件 ...", end=" ", flush=True)
    pcds = []
    for f in mock_files:
        pcd = o3d.io.read_point_cloud(f)
        pcds.append(pcd)
    print("完成")

    idx = [0]  # 当前索引（用 list 以便在闭包中修改）

    def _print_info():
        pcd = pcds[idx[0]]
        bbox = pcd.get_axis_aligned_bounding_box()
        dims = bbox.max_bound - bbox.min_bound
        name = os.path.basename(mock_files[idx[0]])
        print(
            f"  [{idx[0] + 1:>3}/{len(mock_files)}] {name:<55}"
            f"  {dims[0]:>5.0f}×{dims[1]:>5.0f}×{dims[2]:>5.0f}mm"
            f"  {len(pcd.points):>7,} pts"
        )

    def _switch_to(vis, new_idx):
        if new_idx < 0 or new_idx >= len(mock_files):
            return
        idx[0] = new_idx
        vis.clear_geometries()
        vis.add_geometry(pcds[idx[0]])
        _print_info()
        ctrl = vis.get_view_control()
        ctrl.set_zoom(0.8)

    def _next(vis):
        _switch_to(vis, (idx[0] + 1) % len(mock_files))
        return True

    def _prev(vis):
        _switch_to(vis, (idx[0] - 1) % len(mock_files))
        return True

    # 初始显示
    first_name = os.path.basename(mock_files[0])
    print(f"\n浏览 180 个 Mock 点云  →  窗口标题: {first_name}")
    print(f"操作: ← 上一个  → 下一个  Esc 退出\n")
    _print_info()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="[←/→] Mock 点云浏览",
        width=1280,
        height=800,
    )
    vis.add_geometry(pcds[0])
    vis.get_render_option().point_size = 2.0

    # 注册快捷键
    vis.register_key_callback(262, _next)  # Right
    vis.register_key_callback(263, _prev)  # Left
    vis.register_key_callback(265, _next)  # Down
    vis.register_key_callback(266, _prev)  # Up

    vis.run()


def main():
    parser = argparse.ArgumentParser(
        description="Mock 点云预览脚本",
        epilog="示例:\n"
        "  python preview_mock.py --browse\n"
        "  python preview_mock.py --file Mock_rectangular_column_001_high.ply",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", "-f", type=str, help="指定 .ply 文件路径或文件名")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有可用文件")
    parser.add_argument(
        "--all", "-a", action="store_true", help="预览所有类型（各一个样本）"
    )
    parser.add_argument(
        "--browse",
        "-b",
        action="store_true",
        help="交互式浏览所有 Mock 文件（←→键切换）",
    )
    args = parser.parse_args()

    if args.list:
        files = list_files()
        print_file_list(files)
        return

    if args.browse:
        browse_all()
        return

    if args.all:
        preview_all()
        return

    if args.file:
        preview_file(args.file)
        return

    # 默认：预览第一个文件
    files = list_files()
    mock_files = [f for f in files if os.path.basename(f).startswith("Mock_")]
    if mock_files:
        preview_file(mock_files[0])
    elif files:
        preview_file(files[0])
    else:
        print("未找到任何 .ply 文件。")
        print("请先运行: python generate_synthetic_dataset.py")
        print(f"或将真实点云文件放入: {POINT_DIR}")


if __name__ == "__main__":
    main()
