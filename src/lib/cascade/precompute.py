"""
S2 预计算与缓存: S1 推理 + 自适应体素降采样 → 磁盘缓存.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lib.cascade.filter import filter_patch


def build_s2_dataset(s1_model, loader_fn, cache_path, device, console,
                     voxel_mm: float = 40.0):
    """用 S1 预计算全部 patch → 过滤 → 缓存到磁盘."""
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

    if os.path.exists(cache_path):
        console.print(f"  [dim]S2 缓存已存在: {cache_path}[/]")
        return

    loader = loader_fn()
    samples = []
    prog = Progress(
        TextColumn("[bold]S2 预计算[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(), console=console,
    )
    task = prog.add_task("precompute", total=len(loader))
    with torch.no_grad(), prog:
        for pts, feat, lbl, scale in loader:
            pts, lbl, scale = pts.to(device), lbl.to(device), scale.to(device)
            B = pts.shape[0]
            logits = s1_model(pts, scale=None)
            s1_probs = F.softmax(logits, dim=-1)
            for b in range(B):
                p, l, s = pts[b], lbl[b], scale[b].item()
                spb = s1_probs[b]
                kp, ks, kl = filter_patch(p, spb, l, s, voxel_mm=voxel_mm)
                if len(kp) < 32:
                    continue
                samples.append((kp.cpu(), ks.cpu(), kl.cpu(), s))
            prog.update(task, advance=1)
    console.print(f"  [green]S2 数据集: {len(samples)} patches[/]")
    torch.save(samples, cache_path)
    console.print(f"  [dim]缓存已保存: {cache_path}[/]")


def load_s2_loader(cache_path: str, batch_size: int, shuffle: bool = False):
    """从缓存加载 S2 数据集, 返回 DataLoader (变长 padding)."""
    samples = torch.load(cache_path, weights_only=False)

    class _S2Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            pts, s1p, lbl, scale = self.data[idx]
            return pts, s1p, lbl, scale

    def collate_fn(batch):
        max_n = max(b[0].shape[0] for b in batch)
        B = len(batch)
        device = batch[0][0].device
        pts_pad = torch.zeros(B, max_n, 3, device=device)
        s1p_pad = torch.zeros(B, max_n, 3, device=device)
        lbl_pad = torch.full((B, max_n), -1, dtype=torch.long, device=device)
        scale_bt = torch.zeros(B, device=device)
        for i, (p, s, l, sc) in enumerate(batch):
            n = p.shape[0]
            pts_pad[i, :n] = p.to(device)
            s1p_pad[i, :n] = s.to(device)
            lbl_pad[i, :n] = l.to(device)
            scale_bt[i] = sc
        return pts_pad, s1p_pad, lbl_pad, scale_bt

    dataset = _S2Dataset(samples)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=0,
    )
