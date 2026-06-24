"""
快速验证：遍历所有构件类型，确保生成正常。
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from lib.pointcloud_generator._builder import BUILDERS, make_point_cloud

print(f"{'构件类型':25s} {'总点数':>8s}  {'混凝土':>8s}  {'钢筋':>6s}  {'套筒':>6s}")
print("-" * 60)

for ctype in BUILDERS:
    rng = np.random.default_rng(42)
    r = make_point_cloud(ctype, density=0.005, rng=rng)
    pts, lbls = r["points"], r["labels"]
    c = {k: int((lbls == k).sum()) for k in range(4)}
    print(f"{ctype:25s} {len(pts):>8d}  {c[1]:>8d}  {c[2]:>6d}  {c[3]:>6d}")

print("\n所有构件类型生成成功 ✓")
