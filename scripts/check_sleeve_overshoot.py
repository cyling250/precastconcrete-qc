"""批量检查所有 sleeve 样本是否超出混凝土边界。"""
import numpy as np, glob

files = sorted(glob.glob('mock/origin/*sleeve*high_clean.npz'))
print(f'检查 {len(files)} 个 sleeve 文件...')
problems = []

for f in files:
    data = np.load(f)
    pts, lbls = data['points'], data['labels']
    sleeve_mask = lbls == 3
    con_mask = lbls == 1
    if sleeve_mask.sum() == 0 or con_mask.sum() == 0:
        continue
    sleeve_pts, con_pts = pts[sleeve_mask], pts[con_mask]
    
    overshoot = np.maximum(0, np.maximum(
        con_pts.min(0) - sleeve_pts.min(0),
        sleeve_pts.max(0) - con_pts.max(0)
    ))
    max_os = overshoot.max()
    
    if max_os > 0.1:
        problems.append((f, overshoot))
        status = '❌'
    else:
        status = '✅'
    
    # 提取构件名
    name = f.split('Mock_')[1].rsplit('_sleeve', 1)[0] if 'Mock_' in f else f
    print(f'{status} {name}: X={overshoot[0]:.1f} Y={overshoot[1]:.1f} Z={overshoot[2]:.1f}')

print(f'\n总计: {len(files)} 文件, {len(problems)} 个超出')
if problems:
    for f, o in problems[:5]:
        print(f'  超出: {f} → {o}')
else:
    print('✅ 所有样本套筒完全在混凝土范围内!')
