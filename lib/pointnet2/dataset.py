"""
Mock 钢筋混凝土点云数据集.

数据已由 DatasetGenerator 按构件类型分层拆分为 train/val/test 三部分,
分别位于 mock/inuse/{train,val,test}/ 目录下.

训练集: GpuCachedTrainLoader — 一次性生成全部 patches → GPU 缓存 → 每 epoch 显存中打乱索引
验证/测试集: RandomValDataset — 纯随机采样, 一次性生成固定不变

标签: 0=噪声(蓝), 1=混凝土(灰), 2=钢筋(红), 3=套筒(黄)
"""

import os, glob, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree


# ── 常量 ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(_PROJECT_ROOT, "mock", "inuse", "train")
VAL_DIR = os.path.join(_PROJECT_ROOT, "mock", "inuse", "val")
TEST_DIR = os.path.join(_PROJECT_ROOT, "mock", "inuse", "test")
PATCH_RADIUS_MM = 150.0  # 15cm 球查询半径 (物理空间 mm)
NUM_POINTS = 8192


def _list_npz_files(directory: str) -> list[str]:
    """列出目录下所有 NPZ 文件."""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not files:
        raise FileNotFoundError(f"未找到 NPZ 文件: {directory}")
    return files


def _load_geo_features(npz_path: str, num_points: int) -> np.ndarray:
    """加载几何特征, 不存在时返回全零 (3通道)."""
    geo_path = npz_path.replace(".npz", "_geo.npy")
    if os.path.exists(geo_path):
        return np.load(geo_path).astype(np.float32)
    return np.zeros((num_points, 3), dtype=np.float32)


# ═══════════════════════════════════════════════════
#  训练集: GPU 缓存加载器
# ═══════════════════════════════════════════════════


class GpuCachedTrainLoader:
    """训练数据一次性生成并缓存在 GPU, 每 epoch 在显存中打乱索引.

    消除 producer 进程和 epoch 间等待, 数据驻留 GPU 显存不占系统 RAM.
    所有随机操作都使用确定性种子, 确保可复现.
    """

    def __init__(
        self,
        file_paths: list,
        num_points: int = 8192,
        batch_size: int = 8,
        keep_ratio: float = 0.4,
        seed: int = 42,
        device: torch.device = None,
        pos_ratio: float = 0.8,
        samples_per_epoch: int = None,
    ):
        self.num_points = num_points
        self.batch_size = batch_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._samples_per_epoch = samples_per_epoch

        rng = np.random.default_rng(seed)

        # ── 1. 加载全部文件 + 建 KDTree ──
        from rich.progress import Progress

        _t0 = time.time()
        cache = []
        for f in file_paths:
            data = np.load(f)
            pts = data["points"].astype(np.float32)
            lbl = data["labels"].astype(np.int64)
            geo = _load_geo_features(f, len(pts))
            tree = cKDTree(pts)
            cache.append({
                "points": pts, "geo": geo, "labels": lbl,
                "tree": tree, "basename": os.path.basename(f), "N": len(pts),
            })
        n_total = sum(c["N"] for c in cache)
        print(f"  [加载] {len(cache)} 文件已缓存 (≈{n_total/1e6:.1f}M 点, {time.time() - _t0:.0f}s)")

        # ── 2. 生成 patches (带进度条) ──
        _t0 = time.time()
        all_patches = []
        file_order = rng.permutation(len(cache))

        with Progress() as progress:
            file_task = progress.add_task("[cyan]生成 patches", total=len(file_order))

            for fi in file_order:
                c = cache[fi]
                pts, geo, lbl, tree = c["points"], c["geo"], c["labels"], c["tree"]
                available = np.ones(c["N"], dtype=bool)
                n_fail = 0
                n_from_this_file = 0

                while available.sum() >= num_points:
                    if n_fail >= 50:
                        break
                    avail_indices = np.where(available)[0]
                    seed_p = rng.choice(avail_indices)
                    ball_idx = tree.query_ball_point(pts[seed_p], r=PATCH_RADIUS_MM)
                    candidates = np.array(
                        [i for i in ball_idx if available[i]], dtype=np.int64
                    )
                    if len(candidates) >= num_points:
                        chosen = rng.choice(candidates, num_points, replace=False)
                    else:
                        chosen_set = set(candidates.tolist())
                        k_needed = min(num_points * 2, len(avail_indices))
                        _, knn_idx = tree.query(pts[seed_p].reshape(1, -1), k=k_needed)
                        for i in knn_idx[0]:
                            if available[i]:
                                chosen_set.add(int(i))
                                if len(chosen_set) >= num_points:
                                    break
                        chosen = np.array(
                            list(chosen_set)[:num_points], dtype=np.int64
                        )
                    if len(chosen) < num_points:
                        n_fail += 1
                        continue
                    n_fail = 0
                    available[chosen] = False

                    # 归一化到单位球
                    patch_pts = pts[chosen]
                    center = patch_pts.mean(axis=0)
                    patch_pts = patch_pts - center
                    scale_mm = np.max(np.linalg.norm(patch_pts, axis=1))
                    patch_pts = patch_pts / (scale_mm + 1e-8)

                    n_pos = (lbl[chosen] == 2).sum() + (lbl[chosen] == 3).sum()
                    all_patches.append(
                        (patch_pts, geo[chosen], lbl[chosen], scale_mm, n_pos / num_points)
                    )
                    n_from_this_file += 1

                progress.update(
                    file_task, advance=1,
                    description=f"[cyan]{c['basename'][:40]} ({n_from_this_file} patches)",
                )

        # ── 3. 筛选: 按 pos_ratio 控制正样本patch比例 ──
        # pos_ratio=0.4 表示含钢筋/套筒的 patch 占最终训练集的 40%
        rng.shuffle(all_patches)
        has_pos = [p for p in all_patches if p[4] > 0]
        no_pos = [p for p in all_patches if p[4] == 0]
        n_concrete_target = max(0, min(len(no_pos), int(len(has_pos) / pos_ratio - len(has_pos))))
        filtered = has_pos + no_pos[:n_concrete_target]
        rng.shuffle(filtered)
        self.N_per_epoch = len(filtered)

        # 打印分布统计
        pos_ratios = np.array([p[4] for p in all_patches])
        sorted_ratios = np.sort(pos_ratios)[::-1]
        print(
            f"  [生成] {len(all_patches)} 总patches → {self.N_per_epoch} 训练patches "
            f"({len(has_pos)} 含钢筋/套筒 + {n_concrete_target} 纯混凝土, "
            f"ratio={len(has_pos)/max(1,self.N_per_epoch):.2f}, "
            f"{time.time()-_t0:.0f}s)"
        )
        print(
            f"  [分布] max={sorted_ratios[0]:.3f} "
            f"P90={sorted_ratios[max(0,int(len(sorted_ratios)*0.1))]:.3f} "
            f"P75={sorted_ratios[max(0,int(len(sorted_ratios)*0.25))]:.3f} "
            f"P50={sorted_ratios[max(0,int(len(sorted_ratios)*0.5))]:.3f}"
        )

        # ── 4. 转换到 GPU tensor ──
        _t0 = time.time()
        self.pts = torch.stack([torch.from_numpy(p[0]) for p in filtered]).to(device)
        self.geo = torch.stack([torch.from_numpy(p[1]) for p in filtered]).to(device)
        self.lbl = torch.stack([torch.from_numpy(p[2]) for p in filtered]).to(device)
        self.scale = torch.tensor([p[3] for p in filtered], dtype=torch.float32, device=device)
        mb = (self.pts.numel() + self.geo.numel() + self.lbl.numel()) * 4 / 1024 / 1024
        print(f"  [GPU] {self.N_per_epoch} patches {mb:.0f}MB → {device} ({time.time()-_t0:.0f}s)")

    def __iter__(self):
        n_use = self._samples_per_epoch if self._samples_per_epoch else self.N_per_epoch
        n_use = min(n_use, self.N_per_epoch)
        indices = torch.randperm(self.N_per_epoch, device=self._device)[:n_use]
        for i in range(0, n_use, self.batch_size):
            idx = indices[i: i + self.batch_size]
            yield (self.pts[idx], self.geo[idx], self.lbl[idx], self.scale[idx])

    def __len__(self):
        n_use = self._samples_per_epoch if self._samples_per_epoch else self.N_per_epoch
        n_use = min(n_use, self.N_per_epoch)
        return (n_use + self.batch_size - 1) // self.batch_size


# ═══════════════════════════════════════════════════
#  验证/测试集: 纯随机采样 (保持原始分布)
# ═══════════════════════════════════════════════════


class RandomValDataset(Dataset):
    """验证/测试集: 邻域采样 (ball query + KNN), 不做钢筋筛选.

    一次性生成, 固定不变 (epoch 间一致, 公平对比).
    """

    def __init__(
        self,
        split: str = "val",
        num_points: int = 8192,
        seed: int = 42,
    ):
        assert split in ("val", "test"), "RandomValDataset 用于 val 或 test"
        data_dir = VAL_DIR if split == "val" else TEST_DIR
        file_paths = _list_npz_files(data_dir)
        self.num_points = num_points

        rng = np.random.default_rng(seed)
        self.patches = []
        for f in file_paths:
            data = np.load(f)
            pts = data["points"].astype(np.float32)
            lbl = data["labels"].astype(np.int64)
            geo = _load_geo_features(f, len(pts))
            tree = cKDTree(pts)
            N = len(pts)
            available = np.ones(N, dtype=bool)
            n_fail = 0

            while available.sum() >= num_points:
                if n_fail >= 50:
                    break
                avail_indices = np.where(available)[0]
                seed_p = rng.choice(avail_indices)
                ball_idx = tree.query_ball_point(pts[seed_p], r=PATCH_RADIUS_MM)
                candidates = np.array(
                    [i for i in ball_idx if available[i]], dtype=np.int64
                )
                if len(candidates) >= num_points:
                    chosen = rng.choice(candidates, num_points, replace=False)
                else:
                    chosen_set = set(candidates.tolist())
                    k_needed = min(num_points * 2, len(avail_indices))
                    _, knn_idx = tree.query(pts[seed_p].reshape(1, -1), k=k_needed)
                    for i in knn_idx[0]:
                        if available[i]:
                            chosen_set.add(int(i))
                            if len(chosen_set) >= num_points:
                                break
                    chosen = np.array(
                        list(chosen_set)[:num_points], dtype=np.int64
                    )
                if len(chosen) < num_points:
                    n_fail += 1
                    continue
                n_fail = 0
                available[chosen] = False

                # 归一化
                patch_pts = pts[chosen]
                center = patch_pts.mean(axis=0)
                patch_pts = patch_pts - center
                scale = np.max(np.linalg.norm(patch_pts, axis=1))
                patch_pts = patch_pts / (scale + 1e-8)
                self.patches.append((patch_pts, geo[chosen], lbl[chosen], scale))

        print(
            f"[RandomValDataset] {split}: {len(self.patches)} patches "
            f"(来自 {len(file_paths)} 文件, 一次性生成固定不变)"
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        pts, geo, lbl, scale = self.patches[idx]
        return (
            torch.from_numpy(pts),
            torch.from_numpy(geo),
            torch.from_numpy(lbl),
            torch.tensor(scale, dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════════


def create_dataloaders(
    num_points: int = 8192,
    batch_size: int = 8,
    keep_ratio: float = 0.4,
    seed: int = 42,
    file_filter: callable = None,
    only_test: bool = False,
    pos_ratio: float = 0.8,
    samples_per_epoch: int = None,
):
    """返回 (make_train_loader, make_val_loader, make_test_loader).

    训练集: GpuCachedTrainLoader — 一次性生成全部 patches, GPU 缓存, 每 epoch 显存打乱.
    验证/测试集: RandomValDataset — 纯随机采样, 一次性生成固定不变.

    Args:
        file_filter: 可选过滤函数, 接收文件路径返回 bool (用于鲁棒性实验).
        only_test: 若 True, 仅创建测试集 (评估已保存模型时节省内存).
        pos_ratio: 含钢筋/套筒的 patch 在训练集中占比目标 (默认 0.8).
        samples_per_epoch: 每 epoch 随机采样的 patch 数, None 表示全部使用.
    """
    train_files = _list_npz_files(TRAIN_DIR) if not only_test else []
    val_files = _list_npz_files(VAL_DIR) if not only_test else []
    test_files = _list_npz_files(TEST_DIR)
    if file_filter is not None:
        train_files = sorted(f for f in train_files if file_filter(f))
        val_files = sorted(f for f in val_files if file_filter(f))
        test_files = sorted(f for f in test_files if file_filter(f))

    print(f"[DataLoader] 训练: {len(train_files)} 文件 | "
          f"验证: {len(val_files)} 文件 | 测试: {len(test_files)} 文件")

    if only_test:
        # 仅创建测试集 (评估已保存模型)
        test_dataset = RandomValDataset(split="test", num_points=num_points, seed=seed + 200)

        def make_test_loader():
            return DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )

        def make_dummy():
            raise RuntimeError("only_test 模式下未创建该 DataLoader")

        return make_dummy, make_dummy, make_test_loader

    # 训练: GPU 缓存 (一次性生成)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = GpuCachedTrainLoader(
        train_files, num_points, batch_size,
        keep_ratio=keep_ratio, seed=seed, device=device,
        pos_ratio=pos_ratio, samples_per_epoch=samples_per_epoch,
    )

    # 验证/测试: RandomValDataset (固定不变)
    val_dataset = RandomValDataset(split="val", num_points=num_points, seed=seed + 100)
    test_dataset = RandomValDataset(split="test", num_points=num_points, seed=seed + 200)

    def make_train_loader():
        return train_loader

    def make_val_loader():
        return DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    def make_test_loader():
        return DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    return make_train_loader, make_val_loader, make_test_loader
