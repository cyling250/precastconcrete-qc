"""检查 PointNeXt 实验训练状态和最新 checkpoints。"""
import json
import os

hist_path = "experiments/checkpoints/baseline_pointnext/history_baseline_pointnext.json"
summ_path = "experiments/checkpoints/baseline_pointnext/summary_baseline_pointnext.json"
best_path = "experiments/checkpoints/baseline_pointnext/best_baseline_pointnext.pth"

print("=" * 60)
print("PointNeXt 实验训练状态")
print("=" * 60)

# summary
if os.path.exists(summ_path):
    s = json.load(open(summ_path))
    print(f"\n📋 summary.json:")
    for k, v in s.items():
        print(f"  {k}: {v}")

# history
if os.path.exists(hist_path):
    hist = json.load(open(hist_path))
    tl = hist.get("train_loss", [])
    vl = hist.get("val_loss", [])
    vm = hist.get("val_miou", [])
    print(f"\n📈 history:")
    print(f"  总训练 epoch 记录数: {len(tl)}")
    print(f"  总验证 epoch 记录数: {len(vl)}")
    if vm:
        best = max(vm, key=lambda x: x["value"])
        print(f"  最佳 mIoU: epoch {best['epoch']} = {best['value']:.4f}")
        print(f"  最近 5 个 epoch:")
        for e in vm[-5:]:
            print(f"    epoch {e['epoch']}: loss={e.get('loss', '?'):.4f}  mIoU={e['value']:.4f}")
else:
    print("\n⚠️  history 文件不存在")

# best model
if os.path.exists(best_path):
    size_mb = os.path.getsize(best_path) / 1024 / 1024
    print(f"\n💾 best checkpoint: {best_path} ({size_mb:.1f} MB)")
else:
    print(f"\n⚠️  best checkpoint 不存在")
