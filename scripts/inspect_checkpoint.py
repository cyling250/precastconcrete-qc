"""读取 checkpoint 的 val_ious 和 val_prf"""
import torch

ckpt = torch.load(r'F:\sz-subway\precastconcrete-qc\checkpoints\best_pointnet2.pth',
                  map_location='cuda', weights_only=False)

print(f"最佳 epoch: {ckpt['epoch']}")
print(f"val_acc: {ckpt['val_acc']:.4f}")

print("\n=== IoU ===")
for k, v in ckpt['val_ious'].items():
    print(f"  {k}: {v:.4f}")

print("\n=== PRF ===")
for name, m in ckpt['val_prf'].items():
    print(f"  {name}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
