# -*- coding: utf-8 -*-
"""
E3 机制实验: 逐级最远点采样 (FPS) 的逐类存活率.

问题: PointNeXt 在 Table 5/6 上输给 PointNeXt++ 的异常, 是否源于
分层 FPS 在极端类不平衡 (混凝土 ~87%, 钢筋+套筒 ~4%) 下丢弃稀疏类?

方法:
  - 对三个骨干 (pointmlp / pointnet++ / pointnext) 的每一次 FPS 调用做
    无侵入记录 (monkeypatch 模块级 farthest_point_sample).
  - FPS 只依赖坐标, 与权重无关, 因此随机初始化权重即可.
  - 在测试划分的 252 个 patch 上, 统计每一级采样后各类点的存活率:
      存活率(class, stage) = 采样后该类点数 / 采样前该类点数
  - 多种子重复 (FPS 起点是随机点), 报告均值.

用法:
  python -m experiments.experiment_e3_fps_survival --out out/E3
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lib.utils import TEST_DIR, console
from lib.utils.datasets import RandomValDataset

CLASS_NAMES = {0: "noise", 1: "concrete", 2: "rebar", 3: "sleeve"}


class FPSRecorder:
    """monkeypatch farthest_point_sample, 按调用顺序记录每级保留的索引."""

    def __init__(self):
        self.calls = []          # [(module_name, N_in, npoint, idx(B,S))]
        self.active = False

    def wrap(self, module, attr):
        orig = getattr(module, attr)
        rec = self

        def wrapper(xyz, npoint):
            idx = orig(xyz, npoint)
            if rec.active:
                rec.calls.append((module.__name__, xyz.shape[1], npoint, idx.detach().cpu()))
            return idx

        setattr(module, attr, wrapper)
        return orig


def survival_for_patch(labels: np.ndarray, calls) -> dict:
    """一次前向的各级存活率: {(N_in, npoint): {class: kept/in}}."""
    out = {}
    y = torch.as_tensor(labels, dtype=torch.long)
    for (_mod, n_in, npoint, idx) in calls:
        # idx: (B, S) — 逐 batch 累计
        key = (n_in, npoint)
        agg = out.setdefault(key, {c: [0, 0] for c in range(4)})
        for b in range(idx.shape[0]):
            kept = y[idx[b]]
            for c in range(4):
                in_cnt = int((y == c).sum())
                kp_cnt = int((kept == c).sum())
                agg[c][0] += kp_cnt
                agg[c][1] += in_cnt
    return out


def run_arch(arch: str, dataset, seeds, device):
    """对一个骨干跑多种子存活率统计."""
    from experiments._s1_trainer import make_s1_model

    import lib.backbones._base as base_mod

    model = make_s1_model(arch).to(device).eval()
    # 各骨干的 FPS 都绑定在各自模块命名空间 / _base 命名空间
    targets = [base_mod]
    modname = f"lib.backbones.{arch.replace('++', 'pp').replace('-', '_')}"
    try:
        targets.append(__import__(modname, fromlist=["x"]))
    except ImportError:
        pass

    rec = FPSRecorder()
    origs = [(m, rec.wrap(m, "farthest_point_sample")) for m in targets]

    # stage 键在三种骨干间统一按 (N_in, npoint) 记
    per_stage = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # stage -> class -> [kept, in]
    per_seed = []
    with torch.no_grad():
        for seed in seeds:
            torch.manual_seed(seed)
            per_stage.clear()
            for i in range(len(dataset)):
                pts, _feat, lbl, _scale = dataset[i]
                xyz = pts.unsqueeze(0).to(device).float()
                y = lbl.numpy() if torch.is_tensor(lbl) else np.asarray(lbl)
                rec.calls = []
                rec.active = True
                model(xyz, scale=None)
                rec.active = False
                sp = survival_for_patch(y, rec.calls)
                for key, cls_dict in sp.items():
                    for c, (kp, inn) in cls_dict.items():
                        per_stage[key][c][0] += kp
                        per_stage[key][c][1] += inn
            snap = {
                f"{n_in}->{npoint}": {
                    CLASS_NAMES[c]: (v[c][0] / max(v[c][1], 1))
                    for c in sorted(v)
                }
                for (n_in, npoint), v in sorted(per_stage.items(), reverse=True)
            }
            per_seed.append(snap)
    for m, orig in origs:
        setattr(m, "farthest_point_sample", orig)
    return per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/E3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1,
                    help="patch 等距抽样步长 (val 缓存含数千 patch, 抽样以控制耗时)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RandomValDataset(split="test", num_points=8192, seed=242)
    if args.stride > 1:
        # patch 列表按文件顺序排列, 等距抽样可覆盖全部 252 个测试文件
        dataset.patches = dataset.patches[:: args.stride]
    console.print(f"[E3] test patches: {len(dataset)}")

    seeds = list(range(42, 42 + args.seeds))
    result = {}
    for arch in ["pointmlp", "pointnet++", "pointnext"]:
        console.rule(f"[E3] survival: {arch}")
        result[arch] = run_arch(arch, dataset, seeds, device)
        console.print_json(json.dumps(result[arch][0], indent=2))

    with open(os.path.join(args.out, "e3_fps_survival.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "n_patches": len(dataset), "result": result},
                  f, indent=2, ensure_ascii=False)
    console.print(f"[green]saved -> {args.out}/e3_fps_survival.json[/]")


if __name__ == "__main__":
    main()
