"""
一次性预计算所有 NPZ 点云的 40mm 邻域几何特征 (线性度/平面度/球度).

输出: 每个 NPZ 对应一个 _geo.npy 文件, shape=(N, 3)
  列 0: 线性度 (linearity)
  列 1: 平面度 (planarity)
  列 2: 球度 (sphericity)

加速:
  - KDTree.query() 批量化 (B=8192) 避免 OOM
  - numpy 向量化 PCA (逐点)
  - 多进程并行 (mp.Pool, 逐文件)
"""

import os, glob, time, multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree

AUGMENTED_DIR = r"f:\sz-subway\precastconcrete-qc\mock\res\point\augmented"
VAL_DIR = r"f:\sz-subway\precastconcrete-qc\mock\res\point\val"
BALL_RADIUS_MM = 40.0
PCA_MIN_POINTS = 12
QUERY_BATCH = 8192


def compute_geo_features_for_file(npz_path: str) -> bool:
    """处理单个 NPZ 文件, 保存 _geo.npy."""
    base = os.path.splitext(npz_path)[0]
    out_path = base + "_geo.npy"

    if os.path.exists(out_path):
        return False

    try:
        data = np.load(npz_path)
        points = data["points"].astype(np.float64)
    except Exception as e:
        print(f"  ✗ 加载失败 {os.path.basename(npz_path)}: {e}")
        return False

    N = len(points)
    tree = cKDTree(points)
    features = np.zeros((N, 3), dtype=np.float32)
    k_neighbors = min(256, N)

    for start in range(0, N, QUERY_BATCH):
        end = min(start + QUERY_BATCH, N)
        batch = points[start:end]
        dist, idx = tree.query(batch, k=k_neighbors, distance_upper_bound=BALL_RADIUS_MM)

        for i_local in range(len(batch)):
            i_global = start + i_local
            valid_mask = dist[i_local, :] < np.inf
            n_valid = valid_mask.sum()
            if n_valid < PCA_MIN_POINTS:
                features[i_global] = [0.0, 0.0, 1.0]
                continue
            neighborhood = points[idx[i_local, valid_mask]]
            cov = np.cov(neighborhood.T)
            λ = np.linalg.eigvalsh(cov)
            denom = λ[2] + 1e-8
            features[i_global] = [(λ[2] - λ[1]) / denom, (λ[1] - λ[0]) / denom, λ[0] / denom]

    np.save(out_path, features)
    return True


def main():
    for label, dir_path in [("augmented", AUGMENTED_DIR), ("val", VAL_DIR)]:
        files = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
        if not files:
            print(f"[{label}] 无 NPZ 文件, 跳过")
            continue
        print(f"[{label}] 总计 {len(files)} 个 NPZ 文件")

        existing = sum(1 for f in files if os.path.exists(f.replace(".npz", "_geo.npy")))
        print(f"[{label}] 已有 {existing} 个 _geo.npy, 待处理 {len(files) - existing} 个")

        to_process = [f for f in files if not os.path.exists(f.replace(".npz", "_geo.npy"))]
        if not to_process:
            print(f"[{label}] 全部已完成!")
            continue

        n_workers = max(1, min(mp.cpu_count() - 1, 8))
        total = len(to_process)
        print(f"[{label}] 启动 {n_workers} 个 worker 并行处理 {total} 个文件...")

        start = time.time()
        with mp.Pool(n_workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(compute_geo_features_for_file, to_process)):
                if (i + 1) % 5 == 0 or i == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    eta = (total - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{total}] {elapsed:.0f}s, "
                          f"{rate:.2f} files/s, ETA {eta:.0f}s")

        elapsed = time.time() - start
        print(f"[{label}] 完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    mp.freeze_support()
    main()
