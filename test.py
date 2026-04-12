import meshio

# 读取你的VTK网格
mesh_concrete = meshio.read(r"output\prefabricated_column\mesh-concrete.vtk")
mesh_raber = meshio.read(r"output\prefabricated_column\mesh-rebar.vtk")
# ======================
# 过滤无效单元（删除 vertex，只保留Abaqus支持的）
# ======================
valid_cells_concrete = []
valid_cells_raber = []
allowed = {"tetra", "hexahedron", "triangle", "quad", "line"}

for block in mesh_concrete.cells:
    if block.type in allowed:
        valid_cells_concrete.append(block)
for block in mesh_raber.cells:
    if block.type in allowed:
        valid_cells_raber.append(block)

# 生成干净网格
clean_mesh_concrete = meshio.Mesh(
    points=mesh_concrete.points, cells=valid_cells_concrete
)
clean_mesh_raber = meshio.Mesh(points=mesh_raber.points, cells=valid_cells_raber)

# 导出 INP
meshio.write("model.inp", clean_mesh_concrete, file_format="abaqus")
meshio.write("model-rebar.inp", clean_mesh_raber, file_format="abaqus")
print("✅ 导出成功！model.inp 可直接导入 Abaqus")
