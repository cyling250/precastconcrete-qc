"""
批量数据集生成器。

DatasetGenerator 将七种构件类型的参数化生成、多级噪声增强、数据集拆分
整合为一个流水线。输出目录结构：

mock/
  origin/          ← 原始无噪声数据 (每种构件 1×)
  add_noise/       ← 综合噪声数据 (3 噪声等级 × 3 密度等级)
  inuse/
    train/         ← 训练集 (6:2:2 拆分)
    val/           ← 验证集
    test/          ← 测试集
"""

import os
import numpy as np
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from ._config import (
    LABEL_CONCRETE, LABEL_REBAR, LABEL_SLEEVE,
    DENSITY_MAP, NUM_VARIANTS_PER_TYPE,
    TRAIN_VAL_TEST_SPLIT,
    MOCK_DIR, ORIGIN_DIR, ADD_NOISE_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
)
from ._noise import apply_noise
from ._builder import BUILDERS, _filter_internal_rebar


# ─── 辅助函数（可序列化，供多进程调用）────────────────────

def _generate_single(args: tuple) -> dict:
    """
    生成单个构件（无噪声）。
    用于多进程。

    Parameters
    ----------
    args : (comp_type, variant_idx, variant, density_name, density_value, seed)
    """
    comp_type, variant_idx, variant, dens_name, dens_val, seed = args
    rng = np.random.default_rng(seed)
    builder = BUILDERS[comp_type]
    result = builder(density=dens_val, rng=rng, variant=variant)
    # 过滤混凝土内部的钢筋点
    result["points"], result["labels"] = _filter_internal_rebar(
        result["points"], result["labels"]
    )
    return {
        "comp_type": comp_type,
        "variant_idx": variant_idx,
        "variant": variant,
        "density": dens_name,
        "points": result["points"],
        "labels": result["labels"],
        "metadata": result["metadata"],
    }


def _generate_and_augment_single(args: tuple) -> list[dict]:
    """
    生成原始数据并添加所有三种噪声等级。
    args: (comp_type, variant_idx, variant, dens_name, dens_val, noise_configs, base_seed)
    """
    comp_type, variant_idx, variant, dens_name, dens_val, noise_configs, base_seed = args
    
    # 生成原始无噪声数据
    rng = np.random.default_rng(base_seed)
    builder = BUILDERS[comp_type]
    result = builder(density=dens_val, rng=rng, variant=variant)
    # 过滤混凝土内部的钢筋点（实际工程中不可见）
    result["points"], result["labels"] = _filter_internal_rebar(
        result["points"], result["labels"]
    )
    
    origin_pts = result["points"].copy()
    origin_labels = result["labels"].copy()
    metadata = result["metadata"]
    
    outputs = []
    
    # 原始数据
    outputs.append({
        "comp_type": comp_type,
        "variant_idx": variant_idx,
        "variant": variant,
        "density": dens_name,
        "noise_level": "clean",
        "points": origin_pts,
        "labels": origin_labels,
        "metadata": metadata,
    })
    
    # 三种噪声等级
    for noise_name, noise_cfg in noise_configs.items():
        rng_noise = np.random.default_rng(base_seed + hash(noise_name) % 10000)
        pts_noisy, lbls_noisy = apply_noise(
            origin_pts, origin_labels, noise_cfg, rng_noise,
        )
        outputs.append({
            "comp_type": comp_type,
            "variant_idx": variant_idx,
            "variant": variant,
            "density": dens_name,
            "noise_level": noise_name,
            "points": pts_noisy,
            "labels": lbls_noisy,
            "metadata": metadata,
        })
    
    return outputs


def _save_npz(output: dict, save_dir: str) -> str:
    """保存单个输出为 NPZ 文件，返回文件路径。"""
    os.makedirs(save_dir, exist_ok=True)
    
    comp_type = output["comp_type"]
    vidx = output["variant_idx"]
    variant = output.get("variant", "rebar")
    dens = output["density"]
    noise = output["noise_level"]
    
    fname = f"Mock_{comp_type}_{vidx:03d}_{variant}_{dens}_{noise}.npz"
    fpath = os.path.join(save_dir, fname)
    
    np.savez_compressed(
        fpath,
        points=output["points"].astype(np.float32),
        labels=output["labels"].astype(np.int32),
    )
    return fpath


# ─── 数据集生成器 ─────────────────────────────────────────

class DatasetGenerator:
    """
    批量数据集生成器。
    
    用法:
        dg = DatasetGenerator(output_dir="mock")
        dg.generate_all(num_workers=4)
    """
    
    def __init__(
        self,
        output_dir: str = MOCK_DIR,
        origin_dir: str = ORIGIN_DIR,
        noise_dir: str = ADD_NOISE_DIR,
        train_dir: str = TRAIN_DIR,
        val_dir: str = VAL_DIR,
        test_dir: str = TEST_DIR,
        num_variants: int = NUM_VARIANTS_PER_TYPE,
        density_levels: Optional[list[str]] = None,
        noise_levels: Optional[list[str]] = None,
        split_ratio: tuple[int, int, int] = TRAIN_VAL_TEST_SPLIT,
    ):
        self.output_dir = output_dir
        self.origin_dir = origin_dir
        self.noise_dir = noise_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_variants = num_variants
        self.density_levels = density_levels or ["high", "medium", "low"]
        self.noise_levels = noise_levels or ["low", "medium", "high"]
        self.split_ratio = split_ratio
        
        self._comp_types = list(BUILDERS.keys())
        from ._config import NOISE_LEVELS
        self._noise_configs = {k: v for k, v in NOISE_LEVELS.items() if k in self.noise_levels}
    
    def generate_all(self, num_workers: int = 4) -> dict:
        """
        完整流水线：生成 → 噪声增强 → 拆分 → 保存。
        
        Returns
        -------
        stats : dict
            {"origin": N_files, "noise": N_files, "train": N_files, "val": N_files, "test": N_files}
        """
        self._ensure_dirs()
        
        print(f"开始批量生成数据集...")
        print(f"  构件类型: {len(self._comp_types)}")
        print(f"  每类变体: {self.num_variants}")
        print(f"  密度等级: {self.density_levels}")
        print(f"  噪声等级: {self.noise_levels}")
        print(f"  训练/验证/测试: {self.split_ratio}")
        print()
        
        # 1) 生成原始+噪声数据
        all_outputs = self._generate_with_noise(num_workers)
        
        # 2) 保存到 origin/ 和 add_noise/
        origin_files, noise_files = self._save_origin_and_noise(all_outputs)
        
        # 3) 拆分到 train/val/test
        train_files, val_files, test_files = self._split_and_save(all_outputs)
        
        stats = {
            "origin": len(origin_files),
            "noise": len(noise_files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }
        
        print(f"\n生成完成:")
        print(f"  origin/    : {stats['origin']} 文件")
        print(f"  add_noise/ : {stats['noise']} 文件")
        print(f"  train/     : {stats['train']} 文件")
        print(f"  val/       : {stats['val']} 文件")
        print(f"  test/      : {stats['test']} 文件")
        
        return stats
    
    def _ensure_dirs(self):
        for d in [self.origin_dir, self.noise_dir, self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(d, exist_ok=True)
    
    def _generate_with_noise(self, num_workers: int) -> list[dict]:
        """生成所有构件 + 2 种变体 (rebar/sleeve) + 所有密度 + 所有噪声等级。"""
        tasks = []
        seed = 42
        
        for comp_type in self._comp_types:
            for vi in range(1, self.num_variants + 1):
                for variant in ["rebar", "sleeve"]:
                    for dens_name in self.density_levels:
                        dens_val = DENSITY_MAP[dens_name]
                        task = (
                            comp_type, vi, variant, dens_name, dens_val,
                            self._noise_configs, seed,
                        )
                        tasks.append(task)
                        seed += 1
        
        all_outputs = []
        
        if num_workers <= 1:
            # 单进程模式：Open3D 不支持 Windows spawn 子进程
            for task in tasks:
                try:
                    outputs = _generate_and_augment_single(task)
                    all_outputs.extend(outputs)
                except Exception as e:
                    print(f"  生成失败 {task[0]} v{task[1]} ({task[2]}): {e}")
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_generate_and_augment_single, t): t for t in tasks}
                for future in as_completed(futures):
                    try:
                        outputs = future.result()
                        all_outputs.extend(outputs)
                    except Exception as e:
                        task_info = futures[future]
                        print(f"  生成失败 {task_info[0]} v{task_info[1]}: {e}")
        
        return all_outputs
    
    def _save_origin_and_noise(self, outputs: list[dict]) -> tuple[list, list]:
        """保存原始和噪声数据到对应目录。"""
        origin_files, noise_files = [], []
        
        for out in outputs:
            if out["noise_level"] == "clean":
                fpath = _save_npz(out, self.origin_dir)
                origin_files.append(fpath)
            else:
                fpath = _save_npz(out, self.noise_dir)
                noise_files.append(fpath)
        
        return origin_files, noise_files
    
    def _split_and_save(self, outputs: list[dict]) -> tuple[list, list, list]:
        """按 6:2:2 拆分并保存到 inuse/ 目录。"""
        # 只拆分有噪声的数据
        noisy_outputs = [o for o in outputs if o["noise_level"] != "clean"]
        
        # 按 (comp_type, variant_idx) 分组
        from collections import defaultdict
        groups = defaultdict(list)
        for out in noisy_outputs:
            key = (out["comp_type"], out["variant_idx"])
            groups[key].append(out)
        
        unique_keys = list(groups.keys())
        rng = np.random.default_rng(42)
        rng.shuffle(unique_keys)
        
        n_total = len(unique_keys)
        n_train = int(n_total * self.split_ratio[0] / sum(self.split_ratio))
        n_val = int(n_total * self.split_ratio[1] / sum(self.split_ratio))
        
        train_keys = set(unique_keys[:n_train])
        val_keys = set(unique_keys[n_train:n_train + n_val])
        test_keys = set(unique_keys[n_train + n_val:])
        
        train_files, val_files, test_files = [], [], []
        
        for out in noisy_outputs:
            key = (out["comp_type"], out["variant_idx"])
            if key in train_keys:
                fpath = _save_npz(out, self.train_dir)
                train_files.append(fpath)
            elif key in val_keys:
                fpath = _save_npz(out, self.val_dir)
                val_files.append(fpath)
            elif key in test_keys:
                fpath = _save_npz(out, self.test_dir)
                test_files.append(fpath)
        
        return train_files, val_files, test_files
