"""
4 分类点云可视化工具: 浏览 mock/origin/ 目录下的 NPZ 文件.

按键:
  A / ←   上一页
  D / →   下一页
  R       重置视角 (从初始视角查看构件)
  Q / ESC 退出

颜色方案:
  噪声(0)     → 浅蓝 (0.6, 0.6, 1.0)
  混凝土(1)   → 浅灰 (0.7, 0.7, 0.7)
  钢筋(2)     → 暗红 (0.8, 0.2, 0.2)
  套筒(3)     → 金黄 (1.0, 0.8, 0.0)
"""

import os
import sys
import glob
import numpy as np
import open3d as o3d

# 颜色映射
COLOR_MAP = {
    0: (0.6, 0.6, 1.0),   # 噪声 → 浅蓝
    1: (0.7, 0.7, 0.7),   # 混凝土 → 浅灰
    2: (0.8, 0.2, 0.2),   # 钢筋 → 暗红
    3: (1.0, 0.8, 0.0),   # 套筒 → 金黄
}
LABEL_NAMES = {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "mock", "add_noise")


def _build_cloud(file_path: str) -> tuple:
    """加载 NPZ, 返回 (o3d.geometry.PointCloud, title_str)."""
    data = np.load(file_path)
    pts = data["points"]
    lbl = data["labels"]
    colors = np.array([COLOR_MAP.get(int(l), (1, 1, 1)) for l in lbl], dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    parts = name_no_ext.split("_")
    n_total = len(pts)
    n_noise = int((lbl == 0).sum())
    n_conc = int((lbl == 1).sum())
    n_rebar = int((lbl == 2).sum())
    n_sleeve = int((lbl == 3).sum())
    info = (
        f"{basename}  |  "
        f"点: {n_total:,}  "
        f"噪声: {n_noise:,} ({n_noise/n_total*100:.1f}%)  "
        f"混凝土: {n_conc:,} ({n_conc/n_total*100:.1f}%)  "
        f"钢筋: {n_rebar:,} ({n_rebar/n_total*100:.1f}%)  "
        f"套筒: {n_sleeve:,} ({n_sleeve/n_total*100:.1f}%)"
    )
    return cloud, info


def browse_origin():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not files:
        print(f"错误: {DATA_DIR} 下没有 NPZ 文件")
        sys.exit(1)

    print(f"共 {len(files)} 个 NPZ 文件")
    print("操作: [A/←] 上一页  [D/→] 下一页  [R] 重置视角  [Q/ESC] 退出")

    idx = 0
    cloud, info = _build_cloud(files[0])
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="4 分类点云浏览", width=1280, height=720)

    def _update(idx):
        nonlocal cloud, info
        vis.clear_geometries()
        cloud, info = _build_cloud(files[idx])
        vis.add_geometry(cloud)
        vis.reset_view_point(True)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = (0.1, 0.1, 0.1)
        ctrl = vis.get_view_control()
        ctrl.set_zoom(0.7)
        print(f"[{idx+1}/{len(files)}] {info}")

    def next_page(vis):
        nonlocal idx
        if idx < len(files) - 1:
            idx += 1
            _update(idx)
        else:
            print("已是最后一个文件")

    def prev_page(vis):
        nonlocal idx
        if idx > 0:
            idx -= 1
            _update(idx)
        else:
            print("已是第一个文件")

    def reset_view(vis):
        nonlocal idx
        _update(idx)

    def exit_app(vis):
        vis.destroy_window()

    vis.register_key_callback(ord("A"), prev_page)
    vis.register_key_callback(ord("D"), next_page)
    vis.register_key_callback(68, next_page)       # 'd'
    vis.register_key_callback(65, prev_page)       # 'a'
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(114, reset_view)     # 'r'
    vis.register_key_callback(ord("Q"), exit_app)
    vis.register_key_callback(81, exit_app)        # 'Q'
    vis.register_key_callback(256, exit_app)       # ESC

    _update(0)
    vis.run()


if __name__ == "__main__":
    browse_origin()
