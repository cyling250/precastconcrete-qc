"""
显存占用核对: 用稿件所述的训练配置 (batch=16, 8192 点/样本) 在本机 GPU 上跑
一次前向+反向, 测量实际显存占用峰值.

背景: 稿件 "Implementation details" 写的是 single NVIDIA RTX 4060Ti with 16 GB of
VRAM. 本机实测为 RTX 4060 Laptop / 8 GB. 该测试用于判断:
  (a) 论文所述 16 GB 是否确实必要 (8 GB 能否复现);
  (b) 复现说明是否需要补充更具体的显存信息.
"""

from __future__ import annotations

import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments._s1_trainer import make_s1_model  # noqa: E402
from experiments._a2_trainer import SingleStageModel  # noqa: E402
from lib.cascade.stage2_model import Stage2Model    # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}")
print(f"total VRAM = {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB\n")


def probe(name: str, build, in_ch: int, n_cls: int, batch: int = 16, n_pts: int = 8192):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = build().to(DEV)
        model.train()
        if in_ch == 6:
            x = torch.randn(batch, n_pts, 3, device=DEV)
            pri = torch.randn(batch, n_pts, 3, device=DEV).softmax(-1)
            logits = model(x, pri, scale=None)
        else:
            x = torch.randn(batch, n_pts, 3, device=DEV)
            logits = model(x, scale=None)
        y = torch.randint(0, n_cls, (batch, n_pts), device=DEV)
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1), y, ignore_index=-1
        )
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        nparam = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[OK ] {name:<34} batch={batch} points={n_pts} "
              f"参数量={nparam:6.2f}M  峰值显存={peak:5.2f} GB")
        del model, logits, loss, x, y
        return peak
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:90]
        print(f"[OOM] {name:<34} batch={batch} points={n_pts}  -> {msg}")
        return None
    finally:
        torch.cuda.empty_cache()
        gc.collect()


probe("S1 PointMLP (3类)", lambda: make_s1_model("pointmlp"), 3, 3)
probe("S1 PointNet (3类)", lambda: make_s1_model("pointnet"), 3, 3)
probe("S1 PointNet++ (3类)", lambda: make_s1_model("pointnet++"), 3, 3)
probe("S1 PointNeXt (3类)", lambda: make_s1_model("pointnext"), 3, 3)
probe("S2 级联精分割 (6通道输入)", Stage2Model, 6, 4, batch=8)
probe("A2 单阶段 4 类 (M3 消融)", lambda: SingleStageModel(4, 3), 3, 4)
probe("A2 单阶段 4 类 (M3 消融)", lambda: SingleStageModel(4, 3), 3, 4, batch=32)
