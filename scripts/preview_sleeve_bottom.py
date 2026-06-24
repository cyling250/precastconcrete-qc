"""
快速预览灌浆套筒底部效果。
加载一个含套筒的 NPZ 文件，分离 sleeve 点云并观察底部。
"""
import numpy as np
import open3d as o3d

# 选择一个含套筒的 L 形柱或矩形柱
import glob
sleeve_files = glob.glob("mock/origin/Mock_l_shaped_column_*sleeve*high*.npz")
if not sleeve_files:
    sleeve_files = glob.glob("mock/origin/Mock_rectangular_column_*sleeve*high*.npz")
print(f"加载: {sleeve_files[0]}")
data = np.load(sleeve_files[0])
pts, lbls = data["points"], data["labels"]

print(f"总点数: {len(pts)}")
unique, counts = np.unique(lbls, return_counts=True)
for u, c in zip(unique, counts):
    name = {0: "noise", 1: "concrete", 2: "rebar", 3: "sleeve"}.get(u, str(u))
    print(f"  标签 {u} ({name}): {c} 点")

# 分离套筒点云
sleeve_mask = lbls == 3
sleeve_pts = pts[sleeve_mask]
print(f"\n套筒点数: {len(sleeve_pts)}")

# 统计每个套筒的底部点数（z 最小的点）
# 套筒沿 X 轴向（梁）或 Z 轴向（柱），底部在端面内侧
# 通过聚类查看每个套筒底部
from sklearn.cluster import DBSCAN

if len(sleeve_pts) > 0:
    # 检查套筒是否超出混凝土
    con_mask = lbls == 1
    con_pts = pts[con_mask]
    c_min = con_pts.min(axis=0)
    c_max = con_pts.max(axis=0)
    s_min = sleeve_pts.min(axis=0)
    s_max = sleeve_pts.max(axis=0)
    overshoot = np.maximum(0, np.maximum(c_min - s_min, s_max - c_max))
    print(f"\n套筒超出混凝土检查: X={overshoot[0]:.1f}mm, Y={overshoot[1]:.1f}mm, Z={overshoot[2]:.1f}mm")
    if overshoot.max() > 0.1:
        print("⚠️  套筒超出混凝土边界!")
    else:
        print("✅ 套筒完全在混凝土范围内")
    
    # 聚类套筒点
    clustering = DBSCAN(eps=20, min_samples=3).fit(sleeve_pts[:, :2])
    n_clusters = clustering.labels_.max() + 1
    print(f"\n套筒数量（聚类）: {n_clusters}")
    
    for cid in range(n_clusters):
        mask = clustering.labels_ == cid
        sp = sleeve_pts[mask]
        # 套筒沿 Z 轴，两端分别在 z_min 和 z_max
        z_min = sp[:, 2].min()
        z_max = sp[:, 2].max()
        # 底部是 z 最小端，顶部是 z 最大端
        # 底部点判定：z 在最小端的 10mm 范围内
        bottom_mask = sp[:, 2] <= z_min + 10
        top_mask = sp[:, 2] >= z_max - 10
        n_bottom = bottom_mask.sum()
        n_top = top_mask.sum()
        n_wall = len(sp) - n_bottom - n_top
        print(f"  套筒 {cid}: {len(sp)} 点 (壁{n_wall} + 底{n_bottom} + 顶{n_top}), z范围 [{z_min:.0f}, {z_max:.0f}]")

# 可视化所有点，套筒标黄色
colors = np.zeros((len(pts), 3))
colors[lbls == 0] = [0.3, 0.3, 1.0]    # 噪声-蓝
colors[lbls == 1] = [0.7, 0.7, 0.7]    # 混凝土-灰
colors[lbls == 2] = [1.0, 0.2, 0.2]    # 钢筋-红
colors[lbls == 3] = [1.0, 0.8, 0.0]    # 套筒-黄

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 仅看套筒附近
sleeve_center = sleeve_pts.mean(axis=0) if len(sleeve_pts) > 0 else np.zeros(3)
bbox = np.ptp(sleeve_pts, axis=0) if len(sleeve_pts) > 0 else np.array([200, 200, 200])
crop_center = sleeve_center
crop_size = bbox.max() + 50

# 裁剪到套筒区域
if len(sleeve_pts) > 0:
    crop_min = sleeve_pts.min(axis=0) - 20
    crop_max = sleeve_pts.max(axis=0) + 20
    crop_bbox = np.array([crop_min, crop_max])
    in_crop = np.all((pts >= crop_min) & (pts <= crop_max), axis=1)
    print(f"\n裁剪区域点数: {in_crop.sum()} / {len(pts)}")
    pcd_cropped = pcd.select_by_index(np.where(in_crop)[0])
else:
    pcd_cropped = pcd

print("\n可视化：黄色=套筒，灰色=混凝土，红色=钢筋，蓝色=噪声")
print("用鼠标旋转/缩放查看套筒底部是否有黄色点")
o3d.visualization.draw_geometries([pcd_cropped], window_name="套筒底部预览", width=1200, height=800)
