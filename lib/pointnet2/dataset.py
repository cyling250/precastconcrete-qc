"""
Mock 钢筋混凝土点云数据集.

数据已由 DatasetGenerator 按构件类型分层拆分为 train/val/test 三部分,
分别位于 mock/inuse/{train,val,test}/ 目录下.

训练集: PatchBasedDataset — 15cm 球查询局部块采样 + 按钢筋比例筛选
    + DoubleBuffer — 生产者进程预生成下一 epoch, 与训练重叠执行
验证/测试集: RandomValDataset — 纯随机采样 (保持分布不变)

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
_MAX_PATCHES = 3000  # 共享内存缓冲区的上界


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
    # 兜底: 返回全零特征
    return np.zeros((num_points, 3), dtype=np.float32)


# ═══════════════════════════════════════════════════
#  训练集: 局部块采样 (每 epoch 重新生成)
# ═══════════════════════════════════════════════════


class PatchBasedDataset(Dataset):
    """每 epoch 重新生成局部块, 按钢筋比例保留前 40%.

    一次性加载全部文件 + 缓存 KDTrees 在内存中,
    每 epoch 仅维护可用点掩码, 避免重复读盘和建树.

    每 epoch 流程:
    0. 文件列表随机打乱
    1. 对每个文件:
       - 用 15cm ball query 从可用点池中抽取 8192 点子集
       - 不足 8192 时用 KNN 补充
       - 标记已用点不可再用, 重复直到点数耗尽
    2. 统计所有块的钢筋比例
    3. 按钢筋比例降序排列, 保留前 keep_ratio (默认 40%)
    4. 打乱后作为本 epoch 的训练数据
    """

    def __init__(
        self,
        split: str = "train",
        num_points: int = 8192,
        keep_ratio: float = 0.4,
        seed: int = 42,
    ):
        assert split == "train", "PatchBasedDataset 仅用于训练"
        self.file_paths = _list_npz_files(TRAIN_DIR)
        self.num_points = num_points
        self.keep_ratio = keep_ratio
        self.patches = []

        # ── 一次性加载全部文件到内存 + 建 KDTree ──
        _t0 = time.time()
        self._cache = []
        for f in self.file_paths:
            data = np.load(f)
            pts = data["points"].astype(np.float32)
            lbl = data["labels"].astype(np.int64)
            geo = _load_geo_features(f, len(pts))
            tree = cKDTree(pts)
            self._cache.append(
                {
                    "points": pts,
                    "geo": geo,
                    "labels": lbl,
                    "tree": tree,
                    "basename": os.path.basename(f),
                    "N": len(pts),
                }
            )
        n_pts = sum(c["N"] for c in self._cache)
        print(
            f"[PatchBasedDataset] train: {len(self._cache)} 文件已缓存 "
            f"(≈{n_pts / 1e6:.1f}M 点, {time.time() - _t0:.0f}s)"
        )

    def _normalize_patch(self, points: np.ndarray) -> tuple:
        center = points.mean(axis=0)
        points = points - center
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / (scale + 1e-8)
        return points, scale

    def regenerate(self):
        """快速重写: 用缓存 tree, 只维护 available 掩码."""
        rng = np.random.default_rng()
        file_order = rng.permutation(len(self._cache))
        all_patches = []

        from rich.progress import Progress

        with Progress() as progress:
            file_task = progress.add_task("[cyan]Files", total=len(file_order))
            patch_task = progress.add_task("[green]  Patches", total=100, visible=False)

            for fi in file_order:
                c = self._cache[fi]
                pts, geo, lbl, tree = c["points"], c["geo"], c["labels"], c["tree"]
                N = c["N"]
                est_patches = N // self.num_points
                progress.update(
                    patch_task,
                    visible=True,
                    total=est_patches,
                    completed=0,
                    description=f"[green]  {c['basename'][:35]}",
                )
                available = np.ones(N, dtype=bool)
                n_patches_this_file = 0
                n_fail_streak = 0

                while available.sum() >= self.num_points:
                    if n_fail_streak >= 50:
                        break
                    avail_indices = np.where(available)[0]
                    seed = rng.choice(avail_indices)

                    ball_idx = tree.query_ball_point(pts[seed], r=PATCH_RADIUS_MM)
                    candidates = np.array(
                        [i for i in ball_idx if available[i]], dtype=np.int64
                    )

                    if len(candidates) >= self.num_points:
                        chosen = rng.choice(candidates, self.num_points, replace=False)
                    else:
                        chosen_set = set(candidates.tolist())
                        k_needed = min(self.num_points * 2, len(avail_indices))
                        _, knn_idx = tree.query(pts[seed].reshape(1, -1), k=k_needed)
                        for i in knn_idx[0]:
                            if available[i]:
                                chosen_set.add(int(i))
                                if len(chosen_set) >= self.num_points:
                                    break
                        chosen = np.array(
                            list(chosen_set)[: self.num_points], dtype=np.int64
                        )

                    if len(chosen) < self.num_points:
                        n_fail_streak += 1
                        continue

                    n_fail_streak = 0
                    available[chosen] = False
                    normed, scale_mm = self._normalize_patch(pts[chosen])
                    n_pos = (lbl[chosen] == 2).sum() + (lbl[chosen] == 3).sum()
                    pos_ratio = n_pos / self.num_points
                    all_patches.append(
                        (normed, geo[chosen], lbl[chosen], scale_mm, pos_ratio)
                    )
                    n_patches_this_file += 1
                    progress.update(patch_task, advance=1)

                progress.update(patch_task, visible=False)
                progress.update(
                    file_task,
                    advance=1,
                    description=f"[cyan]{c['basename'][:40]} ({n_patches_this_file} patches)",
                )

        # ── 筛选: 全保留含钢筋/套筒的, 纯混凝土只留 5 个 ──
        pos_ratios = np.array([p[4] for p in all_patches])
        sorted_ratios = np.sort(pos_ratios)[::-1]
        n_pos = (pos_ratios > 0).sum()
        n_zero = (pos_ratios == 0).sum()
        print(
            f"  [分布] 总patches={len(sorted_ratios)}, "
            f"含钢筋/套筒={n_pos}, "
            f"纯混凝土={n_zero}, "
            f"max={sorted_ratios[0]:.3f} "
            f"P90={sorted_ratios[max(0, int(len(sorted_ratios) * 0.1))]:.3f} "
            f"P75={sorted_ratios[max(0, int(len(sorted_ratios) * 0.25))]:.3f} "
            f"P50={sorted_ratios[max(0, int(len(sorted_ratios) * 0.5))]:.3f} "
            f"min={sorted_ratios[-1]:.3f}"
        )
        n_all = len(sorted_ratios)
        step = max(1, n_all // 100)
        lines = []
        for i in range(0, min(n_all, 500), step):
            lines.append(f"{sorted_ratios[i]:.3f}")
        print(f"  [降序] {' '.join(lines)}")

        rng.shuffle(all_patches)
        has_pos = [p for p in all_patches if p[4] > 0]
        no_pos = [p for p in all_patches if p[4] == 0]
        KEEP_ZERO = min(5, len(no_pos))
        self.patches = has_pos + no_pos[:KEEP_ZERO]
        rng.shuffle(self.patches)

        print(
            f"  ✓ regenerated: {len(all_patches)} → {len(self.patches)} patches "
            f"(含钢筋/套筒={len(has_pos)}, 纯混凝土保留={KEEP_ZERO})"
        )
        return len(self.patches)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        normed, geo, labels, scale_mm, _ = self.patches[idx]
        return (
            torch.from_numpy(normed),
            torch.from_numpy(geo),
            torch.from_numpy(labels),
            torch.tensor(scale_mm, dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════
#  验证/测试集: 纯随机采样 (保持原始分布)
# ═══════════════════════════════════════════════════


class RandomValDataset(Dataset):
    """验证/测试集: 邻域采样 (ball query + KNN), 不做钢筋筛选.

    用于 validate 和 test 阶段:
    - 不按钢筋比例筛选 (保持原始分布)
    - 一次性生成, 固定不变 (epoch 间一致, 公平对比)
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

        # ── 预生成 patches (邻域采样, 不做筛选) ──
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
            f"(邻域采样, 来自 {len(file_paths)} 文件, 不做筛选)"
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
#  双缓冲: 生产者-消费者 (共享内存零拷贝)
# ═══════════════════════════════════════════════════

import multiprocessing as mp
from multiprocessing import shared_memory
import gc

_SHM_PFX = "ptbp_"


def _clean_shm(suffix: str = ""):
    """清理共享内存块. suffix="" 清理全部, 否则只清理带后缀的块."""
    pfx = f"{_SHM_PFX}{suffix}_" if suffix else _SHM_PFX
    for suf in ["pts", "geo", "lbl", "scale", "n"]:
        try:
            shm = shared_memory.SharedMemory(name=f"{pfx}{suf}")
            shm.close()
            shm.unlink()
        except Exception:
            pass


class SharedPatchBuffer:
    """共享内存补丁缓冲区.

    布局 (固定最大 _MAX_PATCHES):
      pts:   (_MAX_PATCHES, 8192, 3) float32
      geo:   (_MAX_PATCHES, 8192, 3) float32
      lbl:   (_MAX_PATCHES, 8192)    int64
      scale: (_MAX_PATCHES,)         float32
      n:     (1,)                    int64  — 实际 patches 数
    """

    def __init__(self, create: bool = False, suffix: str = ""):
        self._pfx = f"{_SHM_PFX}{suffix}_" if suffix else _SHM_PFX
        p = _MAX_PATCHES
        self._sizes = {
            "pts": p * NUM_POINTS * 3 * 4,
            "geo": p * NUM_POINTS * 3 * 4,
            "lbl": p * NUM_POINTS * 8,
            "scale": p * 4,
            "n": 8,
        }
        if create:
            _clean_shm(suffix)
            self._shms = {
                k: shared_memory.SharedMemory(
                    name=f"{self._pfx}{k}", create=True, size=v
                )
                for k, v in self._sizes.items()
            }
        else:
            self._shms = {
                k: shared_memory.SharedMemory(name=f"{self._pfx}{k}")
                for k in self._sizes
            }

        self.pts = np.ndarray(
            (p, NUM_POINTS, 3), dtype=np.float32, buffer=self._shms["pts"].buf
        )
        self.geo = np.ndarray(
            (p, NUM_POINTS, 3), dtype=np.float32, buffer=self._shms["geo"].buf
        )
        self.lbl = np.ndarray(
            (p, NUM_POINTS), dtype=np.int64, buffer=self._shms["lbl"].buf
        )
        self.scale = np.ndarray(
            (p,), dtype=np.float32, buffer=self._shms["scale"].buf
        )
        self.n = np.ndarray((1,), dtype=np.int64, buffer=self._shms["n"].buf)

    def close(self):
        for shm in self._shms.values():
            shm.close()

    def unlink(self):
        for shm in self._shms.values():
            try:
                shm.unlink()
            except Exception:
                pass


class PersistCache:
    """持久化缓存版 regenerate: 加载全部文件 + 建树 + 循环提取 patches.

    作为子进程独立运行, 与主进程共用共享内存.
    """

    def __init__(self, file_paths: list, num_points: int, verbose: bool = False):
        self.num_points = num_points
        self._verbose = verbose
        self._cache = []
        for f in file_paths:
            data = np.load(f)
            pts = data["points"].astype(np.float32)
            lbl = data["labels"].astype(np.int64)
            geo = _load_geo_features(f, len(pts))
            tree = cKDTree(pts)
            self._cache.append(
                {
                    "points": pts,
                    "geo": geo,
                    "labels": lbl,
                    "tree": tree,
                    "basename": os.path.basename(f),
                    "N": len(pts),
                }
            )
        if verbose:
            print(f"[Producer] {len(self._cache)} 文件已缓存 (≈{sum(c['N'] for c in self._cache)/1e6:.1f}M 点)")

    def _normalize_patch(self, points):
        center = points.mean(axis=0)
        points = points - center
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / (scale + 1e-8)
        return points, scale

    def regenerate(self, keep_max_zero: int = 5) -> list:
        """返回 [(pts, geo, lbl, scale_mm, rebar_ratio), ...].

        Args:
            keep_max_zero: 保留的纯混凝土 (rebar_ratio=0) patches 数.
                           设为 -1 表示全部保留 (用于验证集).
        """
        rng = np.random.default_rng()
        file_order = rng.permutation(len(self._cache))
        all_patches = []
        n_total_pts = sum(c["N"] for c in self._cache)

        for fi in file_order:
            c = self._cache[fi]
            pts, geo, lbl, tree = c["points"], c["geo"], c["labels"], c["tree"]
            available = np.ones(c["N"], dtype=bool)
            n_fail_streak = 0

            while available.sum() >= self.num_points:
                if n_fail_streak >= 50:
                    break
                avail_indices = np.where(available)[0]
                seed = rng.choice(avail_indices)
                ball_idx = tree.query_ball_point(pts[seed], r=PATCH_RADIUS_MM)
                candidates = np.array(
                    [i for i in ball_idx if available[i]], dtype=np.int64
                )
                if len(candidates) >= self.num_points:
                    chosen = rng.choice(candidates, self.num_points, replace=False)
                else:
                    chosen_set = set(candidates.tolist())
                    k_needed = min(self.num_points * 2, len(avail_indices))
                    _, knn_idx = tree.query(pts[seed].reshape(1, -1), k=k_needed)
                    for i in knn_idx[0]:
                        if available[i]:
                            chosen_set.add(int(i))
                            if len(chosen_set) >= self.num_points:
                                break
                    chosen = np.array(
                        list(chosen_set)[: self.num_points], dtype=np.int64
                    )
                if len(chosen) < self.num_points:
                    n_fail_streak += 1
                    continue
                n_fail_streak = 0
                available[chosen] = False
                normed, scale_mm = self._normalize_patch(pts[chosen])
                n_pos = (lbl[chosen] == 2).sum() + (lbl[chosen] == 3).sum()
                all_patches.append(
                    (normed, geo[chosen], lbl[chosen], scale_mm, n_pos / self.num_points)
                )
                if len(all_patches) >= _MAX_PATCHES:
                    break
            if len(all_patches) >= _MAX_PATCHES:
                break

        # 筛选: 全保留含钢筋/套筒的, 纯混凝土只留 keep_max_zero 个
        rng.shuffle(all_patches)
        if keep_max_zero < 0:
            result = all_patches
        else:
            has_pos = [p for p in all_patches if p[4] > 0]
            no_pos = [p for p in all_patches if p[4] == 0]
            n_keep = min(keep_max_zero, len(no_pos))
            result = has_pos + no_pos[:n_keep]
            rng.shuffle(result)
        if self._verbose:
            print(
                f"[Producer] regenerated: {len(all_patches)}→{len(result)} patches "
                f"(keep_max_zero={keep_max_zero})"
            )
        return result


def _producer_loop(
    file_paths: list,
    num_points: int,
    event_shm_created: mp.Event,
    event_data_ready: mp.Event,
    event_ack_consuming: mp.Event,
    suffix: str = "",
    verbose: bool = False,
    keep_max_zero: int = 5,
):
    """生产者进程主循环: 缓存 → 循环 regenerate → 写入共享内存."""
    buf = SharedPatchBuffer(create=True, suffix=suffix)
    event_shm_created.set()
    cache = PersistCache(file_paths, num_points, verbose=verbose)

    # 首轮生成 (同步)
    patches = cache.regenerate(keep_max_zero=keep_max_zero)
    _write_patches(buf, patches)
    event_data_ready.set()

    while True:
        event_ack_consuming.wait()
        event_ack_consuming.clear()
        if keep_max_zero >= 0:
            # 训练集: 每 epoch 重新生成 (筛选策略变化)
            event_data_ready.clear()
            patches = cache.regenerate(keep_max_zero=keep_max_zero)
            _write_patches(buf, patches)
        # 验证/测试集 (keep_max_zero < 0): 不重新生成, 直接复用首轮缓存
        event_data_ready.set()


def _write_patches(buf: SharedPatchBuffer, patches: list):
    """将 patches 列表写入共享内存."""
    n = len(patches)
    buf.n[0] = n
    for i, (pts, geo, lbl, scale, _) in enumerate(patches):
        buf.pts[i] = pts
        buf.geo[i] = geo
        buf.lbl[i] = lbl
        buf.scale[i] = scale


class BufferDataset(Dataset):
    """消费者 Dataset: 从共享内存读取预生成的 patches.

    首次构造时等待 data_ready, 然后触发生产者开始生成下一轮.
    """

    def __init__(self, buf: SharedPatchBuffer, event_ready: mp.Event, event_ack: mp.Event):
        event_ready.wait()
        self._buf = buf
        self._N = int(buf.n[0])
        event_ack.set()
        event_ready.clear()

    def __len__(self):
        return self._N

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self._buf.pts[idx]),
            torch.from_numpy(self._buf.geo[idx]),
            torch.from_numpy(self._buf.lbl[idx]),
            torch.tensor(self._buf.scale[idx], dtype=torch.float32),
        )


# ═══════════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════════


def create_dataloaders(
    num_points: int = 8192,
    batch_size: int = 8,
    keep_ratio: float = 0.4,
    use_double_buffer: bool = True,
    only_test: bool = False,
    file_filter: callable = None,
):
    """返回 (make_train_loader, make_val_loader, make_test_loader).

    数据已由 DatasetGenerator 按构件类型分层拆分为 train/val/test,
    直接从 mock/inuse/{train,val,test}/ 目录加载.

    Args:
        use_double_buffer: 若 True, 启动生产者子进程预生成数据.
        only_test: 若 True, 仅启动测试生产者 (评估已保存的模型时使用).
        file_filter: 可选过滤函数, 接收文件路径返回 bool (用于鲁棒性实验).
    """
    train_files = _list_npz_files(TRAIN_DIR)
    val_files = _list_npz_files(VAL_DIR)
    test_files = _list_npz_files(TEST_DIR)
    if file_filter is not None:
        train_files = sorted(f for f in train_files if file_filter(f))
        val_files = sorted(f for f in val_files if file_filter(f))
        test_files = sorted(f for f in test_files if file_filter(f))

    print(f"[DataLoader] 训练: {len(train_files)} 文件 | "
          f"验证: {len(val_files)} 文件 | 测试: {len(test_files)} 文件")

    if not use_double_buffer:
        raise NotImplementedError("非双缓冲模式已废弃, 请使用双缓冲")

    # 清理所有可能残留的共享内存块
    for _sfx in ["", "train", "val", "test"]:
        _clean_shm(_sfx)

    def _start_producer(files: list, suffix: str, keep_zero: int) -> tuple:
        shm_created = mp.Event()
        data_ready = mp.Event()
        ack = mp.Event()

        proc = mp.Process(
            target=_producer_loop,
            args=(files, num_points, shm_created, data_ready, ack),
            kwargs={"suffix": suffix, "verbose": False, "keep_max_zero": keep_zero},
            daemon=True,
        )
        proc.start()
        shm_created.wait()
        buf = SharedPatchBuffer(create=False, suffix=suffix)
        return buf, data_ready, ack, proc

    if only_test:
        test_buf, test_data_ready, test_ack, _ = _start_producer(
            test_files, "test", keep_zero=-1
        )

        def make_dummy_loader():
            raise RuntimeError("仅创建了测试 DataLoader")

        def make_test_loader():
            ds = BufferDataset(test_buf, test_data_ready, test_ack)
            return DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )

        return make_dummy_loader, make_dummy_loader, make_test_loader

    train_buf, train_data_ready, train_ack, _ = _start_producer(
        train_files, "train", keep_zero=5
    )
    train_data_ready.wait()
    print(f"[Train Buffer] 就绪, 首轮 {int(train_buf.n[0])} patches")

    val_buf, val_data_ready, val_ack, _ = _start_producer(
        val_files, "val", keep_zero=-1
    )

    test_buf, test_data_ready, test_ack, _ = _start_producer(
        test_files, "test", keep_zero=-1
    )

    def _make_loader(buf, data_ready, ack, shuffle=True):
        ds = BufferDataset(buf, data_ready, ack)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=True,
        )

    def make_train_loader():
        return _make_loader(train_buf, train_data_ready, train_ack, shuffle=True)

    def make_val_loader():
        return _make_loader(val_buf, val_data_ready, val_ack, shuffle=False)

    def make_test_loader():
        return _make_loader(test_buf, test_data_ready, test_ack, shuffle=False)

    return make_train_loader, make_val_loader, make_test_loader
