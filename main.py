# -*- coding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2026/01/19 11:00:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   项目主入口，支持命令行参数
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d

# 导入项目模块
from split.point_neighborhood_features import (
    calculate_neighborhood_features,
    save_features_to_single_file,
)
from split.k_means import PointCloudKMeans, create_point_cloud_feature_vector
from split.ransac_plane_fitting import RANSACPlaneFitting, visualize_results


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="地铁车站预制构件加工质量检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算点云邻域特征
  python main.py neighbor-features -i mock/res/point/预制叠合梁300x600.ply -o output/features.npy
  
  # 点云聚类
  python main.py cluster -i mock/res/point/预制叠合梁300x600.ply -f output/features.npy -o output/cluster.ply
  
  # 平面拟合
  python main.py plane-fit -i mock/res/point/预制叠合梁300x600.ply -o output/plane.ply
        """,
    )

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 邻域特征计算命令
    neighbor_features_parser = subparsers.add_parser(
        "neighbor-features", help="计算点云邻域特征"
    )
    neighbor_features_parser.add_argument(
        "-i", "--input", required=True, help="输入点云文件路径"
    )
    neighbor_features_parser.add_argument(
        "-o", "--output", required=True, help="输出特征文件路径（.npy格式）"
    )
    neighbor_features_parser.add_argument(
        "-d",
        "--distance",
        type=int,
        nargs="+",
        default=[50],
        help="邻域距离列表（单位：mm），默认：50",
    )

    # 聚类命令
    cluster_parser = subparsers.add_parser("cluster", help="点云聚类分析")
    cluster_parser.add_argument("-i", "--input", required=True, help="输入点云文件路径")
    cluster_parser.add_argument(
        "-f", "--features", required=True, help="特征文件路径（.npy格式）"
    )
    cluster_parser.add_argument(
        "-o", "--output", required=True, help="输出聚类结果文件路径（.ply格式）"
    )
    cluster_parser.add_argument(
        "-k", "--clusters", type=int, default=2, help="聚类数量，默认：2"
    )
    cluster_parser.add_argument(
        "-m",
        "--metric",
        choices=["euclidean", "manhattan", "cosine"],
        default="euclidean",
        help="距离度量方式，默认：euclidean",
    )
    cluster_parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=50,
        help="特征邻域距离（单位：mm），默认：50",
    )

    # 平面拟合命令
    plane_parser = subparsers.add_parser("plane-fit", help="点云平面拟合")
    plane_parser.add_argument("-i", "--input", required=True, help="输入点云文件路径")
    plane_parser.add_argument(
        "-o", "--output", required=True, help="输出拟合结果文件路径（.ply格式）"
    )
    plane_parser.add_argument(
        "-t", "--threshold", type=float, default=0.05, help="距离阈值，默认：0.05"
    )
    plane_parser.add_argument(
        "-c", "--confidence", type=float, default=0.99, help="置信度，默认：0.99"
    )
    plane_parser.add_argument(
        "-m", "--max-iter", type=int, default=5000, help="最大迭代次数，默认：5000"
    )

    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()

    if not args.command:
        print("错误：请指定命令")
        print("使用 --help 查看可用命令")
        return

    # 执行相应的命令
    if args.command == "neighbor-features":
        run_neighbor_features(args)
    elif args.command == "cluster":
        run_cluster(args)
    elif args.command == "plane-fit":
        run_plane_fit(args)
    else:
        print(f"错误：未知命令 '{args.command}'")
        print("使用 --help 查看可用命令")


if __name__ == "__main__":
    main()
