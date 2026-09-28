# -*- coding: utf-8 -*-
"""
E1: 统一 cmIoU 口径后重算手稿 Table 5 / Table 6 / Table 8.

评测协议与训练脚本完全一致 (create_dataloaders only_test, RandomValDataset
seed=242; A1a 用 RandomSampleDataset), 不另写指标代码:
  - S1 行:   validate_s1        → {噪声, 混凝土, 金属}, rsIoU=金属, cm=cm_iou(c,m,m)
  - 管线行:  val_v2_simple      → 4 类 + rsIoU + cm=cm_iou(c,r,s)
  - M3/M4/M5: validate_a2       → 4 类, cm=cm_iou(c,r,s)

产出 out/E1/E1_tables.{json,md}, 新旧数值并列, 附可复现命令.

用法:
  python -m experiments.experiment_e1_tables --out out/E1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments._config import ARCH_REGISTRY
from experiments._s1_trainer import make_s1_model, validate_s1
from experiments._a2_trainer import SingleStageModel, validate_a2
from lib.cascade import Stage2Model, val_v2_simple
from lib.utils import (
    console, DEVICE, CLASS_NAMES, CHECKPOINT_BASE,
    create_dataloaders, load_checkpoint,
)

# 手稿原值 (单次运行, 旧口径), 用于新旧并列核对
MANUSCRIPT = {
    "table5": {
        "pointmlp":   {"noise": 0.819, "concrete": 0.979, "rs": 0.896, "miou": 0.898, "cm": 0.938},
        "pointnet":   {"noise": 0.171, "concrete": 0.881, "rs": 0.325, "miou": 0.459, "cm": 0.603},
        "pointnet++": {"noise": 0.775, "concrete": 0.973, "rs": 0.858, "miou": 0.869, "cm": 0.916},
        "pointnext":  {"noise": 0.708, "concrete": 0.961, "rs": 0.638, "miou": 0.769, "cm": 0.799},
    },
    "table6": {
        "pointmlp":   {"noise": 0.906, "concrete": 0.922, "rebar": 0.944, "sleeve": 0.901, "rs": 0.927, "miou": 0.918, "cm": 0.922},
        "pointnet":   {"noise": 0.277, "concrete": 0.585, "rebar": 0.812, "sleeve": 0.261, "rs": 0.684, "miou": 0.484, "cm": 0.553},
        "pointnet++": {"noise": 0.882, "concrete": 0.897, "rebar": 0.934, "sleeve": 0.870, "rs": 0.909, "miou": 0.896, "cm": 0.900},
        "pointnext":  {"noise": 0.833, "concrete": 0.850, "rebar": 0.882, "sleeve": 0.731, "rs": 0.821, "miou": 0.824, "cm": 0.821},
    },
    "table8": {
        "S1 Baseline":  {"noise": 0.819, "concrete": 0.979, "rs": 0.896, "miou": 0.898, "cm": 0.938},
        "S1+S2":        {"noise": 0.906, "concrete": 0.922, "rebar": 0.944, "sleeve": 0.901, "miou": 0.918, "cm": 0.922},
        "M1":           {"noise": 0.640, "concrete": 0.959, "rs": 0.780, "miou": 0.793, "cm": 0.870},
        "M2":           {"noise": 0.806, "concrete": 0.978, "rs": 0.866, "miou": 0.883, "cm": 0.922},
        "M3":           {"noise": 0.823, "concrete": 0.981, "rebar": 0.929, "sleeve": 0.862, "miou": 0.899, "cm": 0.924},
    },
}


def _load_s1_state(model, arch):
    p = os.path.join(CHECKPOINT_BASE, f"phase1_baseline_{arch}", "best_s1.pth")
    if not os.path.exists(p):
        return False
    state = torch.load(p, map_location=DEVICE)
    model.load_state_dict(state.get("model_state", state))
    return True


def eval_s1_arch(arch, make_test, criterion):
    model = make_s1_model(arch).to(DEVICE)
    if not _load_s1_state(model, arch):
        console.print(f"[red]缺 S1 权重: {arch}[/]")
        return None
    tl, ious, acc, prf, cm = validate_s1(model, make_test, criterion)
    return {
        "noise": ious["噪声"], "concrete": ious["混凝土"], "rs": ious["金属"],
        "miou": ious["mIoU"], "cm": cm, "acc": acc,
    }


def eval_pipeline(arch, make_test):
    s1 = make_s1_model(arch).to(DEVICE)
    if not _load_s1_state(s1, arch):
        console.print(f"[red]缺 S1 权重: {arch}[/]")
        return None
    s1.eval()
    s2 = Stage2Model().to(DEVICE)
    p2 = os.path.join(CHECKPOINT_BASE, f"phase2_baseline_{arch}", "best_s2.pth")
    if not os.path.exists(p2):
        console.print(f"[red]缺 S2 权重: {arch}[/]")
        return None
    s2.load_state_dict(torch.load(p2, map_location=DEVICE)["model_state"])
    s2.eval()
    c_ious, c_prf, c_acc, _s, rs_iou, cm = val_v2_simple(
        s1, s2, make_test, DEVICE, CLASS_NAMES)
    return {
        "noise": c_ious["噪声"], "concrete": c_ious["混凝土"],
        "rebar": c_ious["钢筋"], "sleeve": c_ious["套筒"],
        "rs": rs_iou, "miou": c_ious["mIoU"], "cm": cm, "acc": c_acc,
    }


def eval_s1_ckpt(tag, make_test, criterion, random_sample):
    """M1/M2: S1 形态的消融 checkpoint (A1a 用 RandomSampleDataset 评测)."""
    ckpt = os.path.join(CHECKPOINT_BASE, tag, "best_s1.pth")
    if not os.path.exists(ckpt):
        console.print(f"[red]缺权重: {tag}[/]")
        return None
    model = make_s1_model("pointmlp").to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state.get("model_state", state))
    tl, ious, acc, prf, cm = validate_s1(model, make_test, criterion)
    return {
        "noise": ious["噪声"], "concrete": ious["混凝土"], "rs": ious["金属"],
        "miou": ious["mIoU"], "cm": cm, "acc": acc,
    }


def eval_a2_ckpt(tag, make_test_patch, criterion):
    """M3/M4/M5: 单阶段 4 类 checkpoint."""
    ckpt = os.path.join(CHECKPOINT_BASE, tag, "best.pth")
    if not os.path.exists(ckpt):
        console.print(f"[red]缺权重: {tag}[/]")
        return None
    model = SingleStageModel(num_classes=4, in_channels=3).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state.get("model_state", state))
    tl, ious, acc, prf, cm = validate_a2(model, make_test_patch, criterion)
    return {
        "noise": ious["噪声"], "concrete": ious["混凝土"],
        "rebar": ious["钢筋"], "sleeve": ious["套筒"],
        "miou": ious["mIoU"], "cm": cm, "acc": acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/E1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    # ── Table 5: S1 四骨干 (ball-query patch 评测) ──
    _, _, make_test = create_dataloaders(8192, 16, keep_ratio=0.4, only_test=True)
    table5 = {}
    for arch in ARCH_REGISTRY:
        console.rule(f"[E1] Table 5 S1: {arch}")
        table5[arch] = eval_s1_arch(arch, make_test, criterion)

    # ── Table 6: 全管线四骨干 ──
    table6 = {}
    for arch in ARCH_REGISTRY:
        console.rule(f"[E1] Table 6 pipeline: {arch}")
        table6[arch] = eval_pipeline(arch, make_test)

    # ── Table 8: 消融 ──
    _, _, make_test_rand = create_dataloaders(
        8192, 16, keep_ratio=0.4, only_test=True, random_sample=True)
    table8 = {}
    table8["S1 Baseline"] = table5.get("pointmlp")
    table8["S1+S2"] = table6.get("pointmlp")
    console.rule("[E1] Table 8: M1 (A1a, RandomSampleDataset 评测)")
    table8["M1"] = eval_s1_ckpt("ablation_A1a_pointmlp", make_test_rand, criterion, True)
    console.rule("[E1] Table 8: M2 (A1b)")
    table8["M2"] = eval_s1_ckpt("ablation_A1b_pointmlp", make_test, criterion, False)
    console.rule("[E1] Table 8: M3 (A2)")
    table8["M3"] = eval_a2_ckpt("ablation_A2", make_test, criterion)
    console.rule("[E1] Table 8: M4 (focal)")
    table8["M4_focal"] = eval_a2_ckpt("ablation_A4_focal", make_test, criterion)
    console.rule("[E1] Table 8: M5 (weighted CE)")
    table8["M5_wce"] = eval_a2_ckpt("ablation_A5_wce", make_test, criterion)

    out = {"table5": table5, "table6": table6, "table8": table8,
           "manuscript_old": MANUSCRIPT}
    with open(os.path.join(args.out, "E1_tables.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    console.print(f"[green]saved -> {args.out}/E1_tables.json[/]")


if __name__ == "__main__":
    main()
