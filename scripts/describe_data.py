"""
描述性统计：显示 mock/ 目录下的所有数据文件（NPZ）信息。
"""
import os
import glob
import numpy as np
from collections import defaultdict

MOCK_DIR = "mock"

def scan_npz_files():
    """扫描 mock/ 下所有 .npz 文件。"""
    files = glob.glob(os.path.join(MOCK_DIR, "**", "*.npz"), recursive=True)
    return files

def parse_filename(fname):
    """从文件名解析信息：Mock_{type}_{variant:03d}_{density}_{noise}.npz"""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    # 构件类型可能含多个下划线，跳过 Mock 前缀
    if parts[0] == "Mock":
        parts = parts[1:]
    # 找到 variant 编号（最后一个纯数字段）
    rev = list(reversed(parts))
    noise = rev[0]
    density = rev[1]
    variant = rev[2]
    # 剩下的就是类型
    comp_type = "_".join(reversed(rev[3:]))
    return comp_type, variant, density, noise

def describe():
    files = scan_npz_files()
    if not files:
        print("⚠️  mock/ 目录下没有找到 .npz 文件。")
        return
    
    print("=" * 70)
    print("                预制混凝土构件点云数据集描述性统计")
    print("=" * 70)
    
    # 按目录分组
    dirs = defaultdict(list)
    for f in files:
        d = os.path.dirname(f)
        dirs[d].append(f)
    
    print(f"\n📁 数据目录结构:\n")
    for dpath in sorted(dirs):
        n = len(dirs[dpath])
        rel = os.path.relpath(dpath, MOCK_DIR) if dpath != MOCK_DIR else "."
        print(f"   mock/{rel:30s} → {n:4d} 个 NPZ 文件")
    
    total_files = len(files)
    total_points = 0
    total_labels = []
    
    # 按类型统计
    type_stats = defaultdict(lambda: {"files": 0, "points": 0, "counts": [0, 0, 0, 0]})
    density_stats = defaultdict(lambda: {"files": 0, "points": 0})
    noise_stats = defaultdict(lambda: {"files": 0, "points": 0})
    
    for f in files:
        try:
            data = np.load(f)
            pts = data["points"]
            lbls = data["labels"]
            n = len(pts)
            total_points += n
            
            comp_type, variant, density, noise = parse_filename(f)
            
            type_stats[comp_type]["files"] += 1
            type_stats[comp_type]["points"] += n
            for k in range(4):
                type_stats[comp_type]["counts"][k] += int((lbls == k).sum())
            
            density_stats[density]["files"] += 1
            density_stats[density]["points"] += n
            
            noise_stats[noise]["files"] += 1
            noise_stats[noise]["points"] += n
            
            total_labels.extend(lbls.tolist())
        except Exception as e:
            print(f"  ⚠️  读取失败: {f}: {e}")
    
    total_labels_arr = np.array(total_labels)
    
    print(f"\n📊 总体统计:\n")
    print(f"  总文件数:     {total_files}")
    print(f"  总点数:       {total_points:,}")
    
    label_names = {0: "噪声", 1: "混凝土", 2: "钢筋", 3: "套筒"}
    for k in range(4):
        cnt = int((total_labels_arr == k).sum())
        pct = cnt / len(total_labels_arr) * 100 if len(total_labels_arr) > 0 else 0
        print(f"  标签 {k} ({label_names[k]:s}): {cnt:>8,} 点 ({pct:.2f}%)")
    
    print(f"\n📋 按构件类型统计:\n")
    print(f"  {'构件类型':30s} {'文件数':>6s}  {'总点数':>10s}  {'混凝土':>10s}  {'钢筋':>8s}  {'套筒':>8s}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for ctype in sorted(type_stats):
        s = type_stats[ctype]
        print(f"  {ctype:30s} {s['files']:>5d}  {s['points']:>10,}  {s['counts'][1]:>10,}  {s['counts'][2]:>8,}  {s['counts'][3]:>8,}")
    
    print(f"\n📋 按密度等级统计:\n")
    for dens in sorted(density_stats):
        s = density_stats[dens]
        print(f"  {dens:10s} → {s['files']:4d} 文件, {s['points']:>10,} 点")
    
    print(f"\n📋 按噪声等级统计:\n")
    for noise in sorted(noise_stats):
        s = noise_stats[noise]
        print(f"  {noise:10s} → {s['files']:4d} 文件, {s['points']:>10,} 点")

if __name__ == "__main__":
    describe()
