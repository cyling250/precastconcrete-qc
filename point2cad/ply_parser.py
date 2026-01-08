import open3d as o3d
import numpy as np

# 读取三角形网格（包含面信息）
mesh = o3d.io.read_triangle_mesh("mesh.ply")

# 计算法线以获得更好的光照效果
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

# 可以选择设置网格颜色（可选）
# mesh.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色

# 绘制网格
print("绘制PLY文件中的面...")
print(f"网格包含 {len(mesh.vertices)} 个顶点和 {len(mesh.triangles)} 个面")
o3d.visualization.draw_geometries([mesh])
