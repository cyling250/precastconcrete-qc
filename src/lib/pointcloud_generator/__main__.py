"""
命令行入口：python -m lib.pointcloud_generator [选项]

用法:
  python -m lib.pointcloud_generator              # 生成一个矩形柱点云
  python -m lib.pointcloud_generator --type t_beam  # 指定构件类型
  python -m lib.pointcloud_generator --all          # 遍历所有构件类型
  python -m lib.pointcloud_generator --batch        # 批量生成完整数据集
"""

import argparse
import numpy as np
import os, sys


def main():
    parser = argparse.ArgumentParser(description="预制混凝土构件点云生成器")
    parser.add_argument("--type", type=str, default="rectangular_column",
                        help="构件类型 (默认: rectangular_column)")
    parser.add_argument("--all", action="store_true",
                        help="遍历所有构件类型并预览")
    parser.add_argument("--batch", action="store_true",
                        help="批量生成完整数据集")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="多进程 worker 数 (默认: 4)")
    
    args = parser.parse_args()
    
    if args.batch:
        from ._dataset import DatasetGenerator
        dg = DatasetGenerator()
        stats = dg.generate_all(num_workers=args.num_workers)
        print(f"数据集生成完成!")
        return
    
    from ._builder import BUILDERS, make_point_cloud
    
    if args.all:
        for comp_type in BUILDERS:
            print(f"\n=== {comp_type} ===")
            result = make_point_cloud(comp_type, density=0.01)
            pts, lbls = result["points"], result["labels"]
            unique, counts = np.unique(lbls, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"  标签 {u}: {c} 点")
            print(f"  总计: {len(pts)} 点")
        return
    
    # 单构件
    result = make_point_cloud(args.type, density=0.01)
    pts, lbls = result["points"], result["labels"]
    
    print(f"构件类型: {args.type}")
    print(f"总点数: {len(pts)}")
    
    unique, counts = np.unique(lbls, return_counts=True)
    for u, c in zip(unique, counts):
        name = {0: "noise", 1: "concrete", 2: "rebar", 3: "sleeve"}.get(u, str(u))
        print(f"  标签 {u} ({name}): {c} 点 ({c/len(pts)*100:.1f}%)")
    
    print(f"\n元数据: {result['metadata']}")


if __name__ == "__main__":
    main()
