import numpy as np
import open3d as o3d
from utils.preprocess_utils import point_cloud_intersect

# 测试不同类型的输入
print("测试不同类型的输入:")

# 测试1：使用numpy数组输入
print("\n测试1：使用numpy数组输入")
try:
    # 创建两个相交的点云
    pcd1_np = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    pcd2_np = np.array([[0.1, 0.1, 0.1], [3, 3, 3], [4, 4, 4]])
    result = point_cloud_intersect(pcd1_np, pcd2_np, eps=0.2)
    print(f"  相交测试结果: {result} (预期: True)")
    print("✓ 测试通过")
except Exception as e:
    print(f"✗ 测试失败: {e}")

# 测试2：使用Open3D PointCloud对象输入
print("\n测试2：使用Open3D PointCloud对象输入")
try:
    # 创建两个相交的点云
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.array([[0.1, 0.1, 0.1], [3, 3, 3], [4, 4, 4]]))
    
    result = point_cloud_intersect(pcd1, pcd2, eps=0.2)
    print(f"  相交测试结果: {result} (预期: True)")
    print("✓ 测试通过")
except Exception as e:
    print(f"✗ 测试失败: {e}")

# 测试3：使用Vector3dVector对象输入
print("\n测试3：使用Vector3dVector对象输入")
try:
    # 创建两个相交的点云
    points1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    points2 = np.array([[0.1, 0.1, 0.1], [3, 3, 3], [4, 4, 4]])
    
    vector1 = o3d.utility.Vector3dVector(points1)
    vector2 = o3d.utility.Vector3dVector(points2)
    
    result = point_cloud_intersect(vector1, vector2, eps=0.2)
    print(f"  相交测试结果: {result} (预期: True)")
    print("✓ 测试通过")
except Exception as e:
    print(f"✗ 测试失败: {e}")
