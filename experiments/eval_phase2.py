"""
Phase 2 测试集评估: 加载最佳 checkpoint, 在测试集上跑完整管线.
"""
import os, sys, json
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from _common import console, DEVICE, CHECKPOINT_BASE, create_dataloaders, CLASS_NAMES, set_seed
from experiment_phase2 import Stage2Model, val_v2_simple

CKPT_DIR = os.path.join(CHECKPOINT_BASE, "phase2")
CKPT_PATH = os.path.join(CKPT_DIR, "best_s2.pth")
S1_CKPT = os.path.join(CHECKPOINT_BASE, "phase1", "best_s1.pth")

set_seed(42)
console.rule("[bold]Phase 2 测试集评估[/]")

# 加载 S1
from experiment_phase1 import FixedRadiusStage1
s1_model = FixedRadiusStage1(num_classes=3, in_channels=3).to(DEVICE)
s1_model.load_state_dict(torch.load(S1_CKPT, map_location=DEVICE))
s1_model.eval()
console.print(f"  S1 已加载")

# 加载 S2
s2_model = Stage2Model().to(DEVICE)
sd = torch.load(CKPT_PATH, map_location=DEVICE)
s2_model.load_state_dict(sd["model_state"])
s2_model.eval()
console.print(f"  S2 已加载 (epoch {sd.get('epoch', '?')}, cm={sd.get('cm', 0):.3f})")

# 测试集
_, _, make_test = create_dataloaders(
    8192, 16, keep_ratio=0.4,
)
c_ious, c_prf, c_acc, s_ious, rs_iou, cm = val_v2_simple(s1_model, s2_model, make_test)

cio = " | ".join(f"{k}={v:.3f}" for k, v in c_ious.items() if k != "mIoU")
console.print(f"  [bold]测试集[/]")
console.print(f"  acc={c_acc:.3f} mIoU={c_ious['mIoU']:.3f} | [{cio}]")
console.print(f"  cm={cm:.3f} | rsIoU={rs_iou:.3f}")

cpr = " | ".join(f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}" for k, v in c_prf.items())
console.print(f"  PRF(组合): [{cpr}]")
sio = " | ".join(f"{k}={v:.3f}" for k, v in s_ious.items() if k != "mIoU")
console.print(f"  S2纯: [{sio}] mIoU(S2)={s_ious.get('mIoU', 0):.3f}")

# 保存
summary = {
    "test_acc": c_acc,
    "test_miou": c_ious["mIoU"],
    "test_cm": cm,
    "test_rs_iou": rs_iou,
    **{f"iou/{k}": v for k, v in c_ious.items() if k != "mIoU"},
    **{f"s2pure/{k}": v for k, v in s_ious.items() if k != "mIoU"},
}
with open(os.path.join(CKPT_DIR, "summary_phase2.json"), "w") as f:
    json.dump(summary, f, indent=2)
console.print(f"[bold green]完成![/]")
