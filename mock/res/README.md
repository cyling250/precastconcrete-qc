# 模型资源文件夹说明

本文件夹包含Revit模型转点云流程中的各个阶段的测试用例和资源文件，与 `../Revit2Point.md`文档中描述的转换流程（`.rvt`→`.ifc`→`.obj+.mtl`→`.pcd`）相对应。

## 文件夹结构

```
res/
├── README.md              # 本说明文件
├── ifc/                   # IFC格式模型文件
├── obj/                   # OBJ格式模型文件
│   ├── standard_layer/    # 标准层模型
│   ├── structure/         # 结构模型
│   └── wall_and_beam/     # 墙和梁模型
├── pcd/                   # PCD格式点云文件
└── revit2020/             # Revit 2020格式模型文件
```

## 文件夹说明

### 1. revit2020/

- **用途**：存放原始Revit 2020格式的模型文件（`.rvt`）
- **说明**：这些是从Revit软件创建或导出的原始三维建筑信息模型文件

### 2. ifc/

- **用途**：存放IFC格式的模型文件（`.ifc`）
- **说明**：这些文件是从Revit模型导出的IFC格式文件，用于保留更完整的建筑信息和材质信息
- **生成方式**：参考 `../Revit2Point.md`文档中的「Revit导出IFC文件」步骤

### 3. obj/

- **用途**：存放OBJ格式的模型文件（`.obj`）和对应的材质文件（`.mtl`）
- **子文件夹说明**：
  - `standard_layer/`：标准层模型，包含建筑的标准楼层结构
  - `structure/`：结构模型，主要包含建筑的承重结构部分
  - `wall_and_beam/`：墙和梁模型，专门存放墙体和梁的模型
- **生成方式**：参考 `../Revit2Point.md`文档中的「IFC转OBJ格式」步骤

### 4. pcd/

- **用途**：存放PCD格式的点云文件（`.pcd`）
- **说明**：这些是从OBJ模型转换生成的点云数据文件，用于后续的点云处理和分析
- **生成方式**：参考 `../Revit2Point.md`文档中的「OBJ转点云并保存为PCD格式」步骤

## 文件转换流程

文件在各个文件夹间的转换流程如下：

1. Revit模型导出：`revit2020/(.rvt)` → `ifc/(.ifc)`
2. IFC转OBJ：`ifc/(.ifc)` → `obj/(.obj+.mtl)`
3. OBJ转点云：`obj/(.obj)` → `pcd/(.pcd)`

详细的转换步骤请参考 `../Revit2Point.md`文档。

## 测试点云

生成的PCD点云文件可以使用 `../test.py`工具进行测试。

## 资源下载

通过网盘分享的文件：precastconcrete-qc.mock.res.rar
链接: [https://pan.baidu.com/s/1IlWkHDnlBPdxBeajxfq-WQ?pwd=e17n](https://pan.baidu.com/s/1IlWkHDnlBPdxBeajxfq-WQ?pwd=e17n) 提取码: e17n
--来自百度网盘超级会员v4的分享中

## 注意事项

1. 请确保同一模型在不同格式间的文件名保持一致，以便于追踪和管理
2. 在转换过程中，请确保勾选「包含材质」选项，以保留模型的材质信息
