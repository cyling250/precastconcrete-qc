"""
测试 Phase 1 (S1) → Phase 2 去噪/降采样管线的可行性.

对照组 (教师强制): GT 标签去噪 + 混凝土体素降采样 (3cm).
实验组 (S1 管线):   S1 预测去噪 + 混凝土体素降采样 (3cm).

预期: 实验组金属保留率略低于对照组, 但差距不大.
"""

import os, sys, math
import torch
import torch.nn.functional as F
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from _common import (
    console, DEVICE, CHECKPOINT_BASE, set_seed,
    create_dataloaders, NUM_POINTS, BATCH_SIZE,
)
from experiment_phase1 import FixedRadiusStage1
from lib.pointnet2 import PointNetPlusPlus, count_params

TAG = "phase1"
CKPT_DIR = os.path.join(CHECKPOINT_BASE, TAG)
CKPT_PATH = os.path.join(CKPT_DIR, "best_s1.pth")
VOXEL_MM = 40.0  # 4cm 体素 (mm)

# ── 4 类标签体系 ──
# 0=噪声, 1=混凝土, 2=钢筋, 3=套筒
# S1 3 类: 0=噪声, 1=混凝土, 2=金属 (钢筋+套筒)


def _voxel_downsample(xyz_norm: torch.Tensor, scale_mm: float) -> torch.Tensor:
    """体素降采样: 归一化坐标 → 3cm 体素 → 返回保留索引.

    Args:
        xyz_norm: (N, 3) 归一化到单位球的坐标
        scale_mm: 原始尺度 (mm), 用于将 3cm 映射到归一化空间
    Returns:
        keep_idx: 降采样后保留的索引 (1-D LongTensor)
    """
    if xyz_norm.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=xyz_norm.device)
    vn = VOXEL_MM / scale_mm  # 归一化空间中的体素大小
    vi = torch.floor(xyz_norm / vn).long()
    sft = vi - vi.min(dim=0).values
    mv = sft.max(dim=0).values + 1
    uid = sft[:, 0] * mv[1] * mv[2] + sft[:, 1] * mv[2] + sft[:, 2]
    su, si = torch.sort(uid)
    fi = torch.ones(len(su), dtype=torch.bool, device=xyz_norm.device)
    fi[1:] = su[1:] != su[:-1]
    return si[fi]


def _map_to_s3(lbl_4: torch.Tensor) -> torch.Tensor:
    """4 类标签 → S1 3 类: 钢筋(2)→金属(2), 套筒(3)→金属(2)."""
    lbl = lbl_4.clone()
    lbl[lbl == 3] = 2
    return lbl


@torch.no_grad()
def compute_retention(true_labels, pred_labels, keep_mask):
    """统计过滤后的点保留情况.

    Args:
        true_labels: (N,) GT 4 类标签
        pred_labels: (N,) S1 预测 3 类标签 (或 None 对control)
        keep_mask:   (N,) bool, 哪些点被保留
    Returns:
        dict 统计量
    """
    N = len(true_labels)
    kept = keep_mask.sum().item()
    gt_noise = (true_labels == 0).sum().item()
    gt_concrete = (true_labels == 1).sum().item()
    gt_metal = ((true_labels == 2) | (true_labels == 3)).sum().item()

    kept_noise = ((true_labels == 0) & keep_mask).sum().item()
    kept_concrete = ((true_labels == 1) & keep_mask).sum().item()
    kept_metal = ((true_labels == 2) & keep_mask).sum().item()
    kept_sleeve = ((true_labels == 3) & keep_mask).sum().item()

    # 降采样后混凝土点 + 完整保留的金属点
    conc_after = ((true_labels == 1) & keep_mask).sum().item()
    metal_after = ((true_labels == 2) & keep_mask).sum().item() + \
                  ((true_labels == 3) & keep_mask).sum().item()

    return {
        "N": N,
        "kept": kept,
        "keep_ratio": kept / max(N, 1),
        "gt_noise": gt_noise,
        "gt_concrete": gt_concrete,
        "gt_metal": gt_metal,
        "kept_noise": kept_noise,
        "kept_concrete": conc_after,
        "kept_metal": metal_after,
        "kept_rebar": kept_metal,
        "kept_sleeve": kept_sleeve,
        "noise_leak_ratio": kept_noise / max(kept, 1),       # 泄漏噪声占保留点比例
        "metal_retention": metal_after / max(gt_metal, 1),    # 金属保留率
        "conc_downsample_ratio": conc_after / max(gt_concrete, 1),  # 混凝土降采样率
        "noise_removal_rate": 1.0 - kept_noise / max(gt_noise, 1),  # 噪声去除率
    }


def process_patch_control(pts, lbl_4, scale_mm):
    """对照组: 教师强制 (GT 去噪 + 体素降采样混凝土)."""
    N = len(lbl_4)
    keep = torch.ones(N, dtype=torch.bool, device=pts.device)

    # 1. 删除噪声 (GT label=0)
    is_noise = lbl_4 == 0
    keep[is_noise] = False

    # 2. 体素降采样混凝土 (GT label=1)
    is_concrete = lbl_4 == 1
    if is_concrete.any():
        conc_idx = torch.where(is_concrete)[0]
        conc_pts = pts[conc_idx]
        ds_idx = _voxel_downsample(conc_pts, scale_mm)
        # 保留降采样后的混凝土索引, 移除其余混凝土
        drop_conc = torch.ones(len(conc_idx), dtype=torch.bool, device=pts.device)
        drop_conc[ds_idx] = False
        keep[conc_idx[drop_conc]] = False

    # 3. 金属点全部保留
    return keep


def process_patch_s1(s1_model, pts, lbl_4, scale_mm):
    """实验组: S1 预测去噪 + 体素降采样."""
    N = len(lbl_4)
    keep = torch.ones(N, dtype=torch.bool, device=pts.device)

    # S1 推理
    logits = s1_model(pts.unsqueeze(0), scale=None)  # (1, N, 3)
    s1_pred = logits.argmax(dim=-1).squeeze(0)  # (N,)  {0=噪, 1=砼, 2=金}

    # 1. 删除预测为噪声的点
    is_noise = s1_pred == 0
    keep[is_noise] = False

    # 2. 体素降采样预测为混凝土的点
    is_concrete = s1_pred == 1
    if is_concrete.any():
        conc_idx = torch.where(is_concrete)[0]
        conc_pts = pts[conc_idx]
        ds_idx = _voxel_downsample(conc_pts, scale_mm)
        drop_conc = torch.ones(len(conc_idx), dtype=torch.bool, device=pts.device)
        drop_conc[ds_idx] = False
        keep[conc_idx[drop_conc]] = False

    # 3. 金属点全部保留
    return keep, s1_pred


def print_comparison(method: str, stats: dict):
    """打印单个方法的汇总统计."""
    if not stats:
        return
    n = len(stats["kept"])
    avg_kr = np.mean(stats["keep_ratio"])
    avg_mr = np.mean(stats["metal_retention"])
    avg_nl = np.mean(stats["noise_leak_ratio"])
    avg_nr = np.mean(stats["noise_removal_rate"])
    avg_cr = np.mean(stats["conc_downsample_ratio"])
    total_kept = sum(stats["kept"])
    total_metal = sum(stats["kept_metal"])
    total_noise = sum(stats["kept_noise"])
    total_rebar = sum(stats["kept_rebar"])
    total_sleeve = sum(stats["kept_sleeve"])

    console.print(f"  [bold]{method}[/]")
    console.print(f"    处理后总点数: {total_kept:,} (avg {total_kept/n:.0f}/patch)")
    console.print(f"    金属保留率:       {avg_mr:.4f} (总计 {total_metal:,} 点, 其中钢筋 {total_rebar:,} / 套筒 {total_sleeve:,})")
    console.print(f"    混凝土降采样率:   {avg_cr:.4f}")
    console.print(f"    噪声去除率:       {avg_nr:.4f}")
    console.print(f"    噪声泄漏占比:     {avg_nl:.4f} (泄漏 {total_noise:,} 点)")
    console.print(f"    点保留率:         {avg_kr:.4f}")


def main():
    set_seed(42)
    console.rule("[bold]Phase 1 → Phase 2 管线测试[/]")
    console.print(f"  体素大小: {VOXEL_MM}mm (3cm)")

    # ── 1. 检查 S1 权重 ──
    if not os.path.exists(CKPT_PATH):
        console.print(f"[red]未找到 S1 权重 {CKPT_PATH}, 请先运行 experiment_phase1.py[/]")
        return

    # ── 2. 加载 S1 模型 ──
    s1_model = FixedRadiusStage1(num_classes=3, in_channels=3).to(DEVICE)
    s1_model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    s1_model.eval()
    console.print(f"  [dim]S1 已加载 ({CKPT_PATH})[/]")

    # ── 3. 加载验证集 (少量 patch 即可) ──
    N_TEST = 200  # 测试 200 个 patch
    _, make_val, _ = create_dataloaders(
        NUM_POINTS, BATCH_SIZE, keep_ratio=0.4,
    )
    val_loader = make_val()
    limited_loader = []
    for i, batch in enumerate(val_loader):
        if i * BATCH_SIZE >= N_TEST:
            break
        limited_loader.append(batch)
    console.print(f"  验证集: {min(N_TEST, len(val_loader)*BATCH_SIZE)} patches ({len(limited_loader)} batch)")

    # ── 4. 遍历验证集, 统计两组指标 ──
    control_stats = {k: [] for k in [
        "N", "kept", "keep_ratio", "gt_noise", "gt_concrete", "gt_metal",
        "kept_noise", "kept_concrete", "kept_metal", "kept_rebar", "kept_sleeve",
        "noise_leak_ratio", "metal_retention", "conc_downsample_ratio", "noise_removal_rate",
    ]}
    exp_stats = {k: [] for k in control_stats.keys()}

    # S1 混淆矩阵统计
    s1_total = {"correct": 0, "total": 0}
    s1_confusion = {"noise→noise": 0, "noise→conc": 0, "noise→metal": 0,
                    "conc→conc": 0, "conc→noise": 0, "conc→metal": 0,
                    "metal→metal": 0, "metal→noise": 0, "metal→conc": 0}

    prog = Progress(
        TextColumn("[bold]测试[/]"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    )
    task = prog.add_task("test", total=len(limited_loader))

    with torch.no_grad(), prog:
        for batch in limited_loader:
            pts, feat, lbl_4, scale = batch
            pts, lbl_4, scale = pts.to(DEVICE), lbl_4.to(DEVICE), scale.to(DEVICE)
            B = pts.shape[0]

            for b in range(B):
                p = pts[b]                     # (N, 3)
                l = lbl_4[b]                   # (N,)
                s = scale[b].item()            # mm

                # ── 对照组 (教师强制) ──
                ctrl_keep = process_patch_control(p, l, s)
                cs = compute_retention(l, None, ctrl_keep)
                for k in control_stats:
                    control_stats[k].append(cs[k])

                # ── 实验组 (S1 管线) ──
                exp_keep, s1_pred = process_patch_s1(s1_model, p, l, s)
                es = compute_retention(l, s1_pred, exp_keep)
                for k in exp_stats:
                    exp_stats[k].append(es[k])

                # S1 混淆矩阵 (4 类 vs 3 类)
                s1_lbl = _map_to_s3(l)
                s1_total["total"] += len(s1_lbl)
                s1_total["correct"] += (s1_pred == s1_lbl).sum().item()
                for gt_cls, name in [(0, "noise"), (1, "conc"), (2, "metal"), (3, "metal")]:
                    mask = (l == gt_cls) if gt_cls < 3 else ((l == 2) | (l == 3))
                    gt_name = "noise" if gt_cls == 0 else ("conc" if gt_cls == 1 else "metal")
                    if mask.any():
                        preds = s1_pred[mask]
                        s1_confusion[f"{gt_name}→noise"] += (preds == 0).sum().item()
                        s1_confusion[f"{gt_name}→conc"] += (preds == 1).sum().item()
                        s1_confusion[f"{gt_name}→metal"] += (preds == 2).sum().item()

            prog.update(task, advance=1)

    # ── 5. 输出结果 ──
    console.rule("[bold]结果对比[/]")
    print_comparison("对照组 (GT 教师强制)", control_stats)
    print_comparison("实验组 (S1 管线)", exp_stats)

    # 差异
    console.rule("[bold]实验组 vs 对照组 差异[/]")
    for key, label in [
        ("metal_retention", "金属保留率"),
        ("noise_removal_rate", "噪声去除率"),
        ("noise_leak_ratio", "噪声泄漏占比"),
        ("keep_ratio", "点保留率"),
        ("conc_downsample_ratio", "混凝土降采样率"),
    ]:
        c_avg = np.mean(control_stats[key])
        e_avg = np.mean(exp_stats[key])
        diff = e_avg - c_avg
        console.print(f"  {label}: 对照={c_avg:.4f} 实验={e_avg:.4f} 差值={diff:+.4f}")

    # S1 准确率
    s1_acc = s1_total["correct"] / max(s1_total["total"], 1)
    console.print(f"\n  S1 3 类准确率: {s1_acc:.4f} ({s1_total['correct']}/{s1_total['total']})")
    console.print(f"  S1 混淆矩阵 (行=GT → 列=S1预测):")
    for gt in ["noise", "conc", "metal"]:
        n = s1_confusion[f"{gt}→noise"] + s1_confusion[f"{gt}→conc"] + s1_confusion[f"{gt}→metal"]
        if n == 0:
            continue
        console.print(f"    {gt:>6s}: noise={s1_confusion[f'{gt}→noise']/n:.3f} "
                      f"conc={s1_confusion[f'{gt}→conc']/n:.3f} "
                      f"metal={s1_confusion[f'{gt}→metal']/n:.3f}")

    console.rule("[bold green]测试完成[/]")


if __name__ == "__main__":
    main()
