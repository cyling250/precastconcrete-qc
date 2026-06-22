"""加载训练 checkpoint 并打印训练历史"""
import torch

ckpt = torch.load(r'F:\sz-subway\precastconcrete-qc\checkpoints\best_pointnet2.pth', map_location='cuda', weights_only=False)
hist = ckpt['history']

print('=== 训练历史关键指标 ===')
epochs = hist['epoch']
last_epoch = epochs[-1]

key_map = {
    'train_loss': '训练 Loss',
    'val_loss': '验证 Loss',
    'val_acc': '验证 Acc',
    'clean_miou': '干净 mIoU',
    'iou/噪声': '噪声 IoU',
    'iou/混凝土': '混凝土 IoU',
    'iou/钢筋': '钢筋 IoU',
    'iou/mIoU': '标准 mIoU',
    'prf/钢筋/precision': '钢筋 P',
    'prf/钢筋/recall': '钢筋 R',
    'prf/钢筋/f1': '钢筋 F1',
    'prf/混凝土/precision': '混凝土 P',
    'prf/混凝土/recall': '混凝土 R',
    'prf/噪声/precision': '噪声 P',
    'prf/噪声/recall': '噪声 R',
}

for k, label in key_map.items():
    if k not in hist:
        continue
    v = hist[k]
    if not v or len(v) == 0:
        continue
    last = v[-1]

    # find best epoch
    is_loss = 'loss' in k.lower()
    best_fn = min if is_loss else max
    best = best_fn(v)
    best_idx = v.index(best)
    best_label = f'best={best:.4f}@E{epochs[best_idx]}'

    print(f'  {label:16s}: last={last:.4f}  ({best_label})')

print(f'\n总 epoch 数: {last_epoch}')
best_ep = hist.get('best_epoch', hist.get('epoch'))
if isinstance(best_ep, list):
    best_ep = best_ep[0]
print(f'最佳模型保存于: Epoch {best_ep}')

# mIoU trend
miou = hist.get('iou/mIoU', [])
if miou and len(miou) > 5:
    print(f'\n=== mIoU 趋势 ===')
    for i in range(0, len(miou), max(1, len(miou)//10)):
        print(f'  E{epochs[i]:3d}: mIoU={miou[i]:.4f}')
    print(f'  E{epochs[-1]:3d}: mIoU={miou[-1]:.4f} (最终)')
