"""重新生成完整数据集（包含所有最新修改）。"""
import sys
sys.path.insert(0, ".")

from lib.pointcloud_generator._dataset import DatasetGenerator

# Open3D 不支持 multiprocessing (Windows spawn)，使用 num_workers=1
dg = DatasetGenerator(output_dir="mock")
stats = dg.generate_all(num_workers=1)

print("\n=== 数据集生成完成 ===")
for k, v in stats.items():
    print(f"  {k}: {v} 文件")
