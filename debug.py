import open3d as o3d

# 打开文件并可视化
pcd = o3d.io.read_point_cloud("C:\\sers\\35160\\Desktop\\mesh (1).ply")
# 可视化
o3d.visualization.draw_geometries([pcd])
