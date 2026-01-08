# precastconcrete-qc

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Open3D 0.19.0](https://img.shields.io/badge/open3d-0.19.0-orange.svg)](http://www.open3d.org/)
[![numpy 2.4.0](https://img.shields.io/badge/numpy-2.4.0-green.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/status-development-yellow.svg)]()
[![Platform](https://img.shields.io/badge/platform-windows-blue.svg)]()

## 项目简介

地铁车站预制构件加工质量检测方法研究项目，主要利用**点云技术**和**计算机视觉技术**实现对预制混凝土构件的加工质量进行自动化检测和评估。本项目目前处于开发阶段，旨在为地铁车站建设中的预制构件质量控制提供高效、准确的解决方案。

## 项目背景

在地铁车站等大型基础设施建设中，预制混凝土构件的加工质量直接影响到工程的安全性和可靠性。传统的人工检测方法效率低下、主观性强，难以满足大规模生产和高质量要求。本项目通过引入点云技术和计算机视觉技术，实现对预制构件的三维尺寸、表面质量、安装精度等关键指标的自动化检测。

## 主要功能

- **Revit模型转点云**：将建筑信息模型(BIM)转换为点云数据，用于质量检测的参考标准
- **点云数据处理**：包括点云加载、可视化、滤波、配准等基础操作
- **质量检测算法**：开发针对预制构件的尺寸偏差、表面缺陷等质量指标的检测算法
- **检测结果可视化**：直观展示检测结果和质量评估报告

## 项目结构

```
precastconcrete-qc/
├── .gitignore                 # Git忽略文件配置
├── .vscode/                  # VSCode配置文件
│   ├── launch.json
│   └── settings.json
├── README.md                 # 项目中文说明文档
├── requirements.txt          # 项目依赖库列表
└── mock/                     # 模型和测试资源目录
    ├── Revit2Point.md        # Revit模型转点云教程
    ├── res/                  # 测试资源
    │   ├── README.md         # 资源说明
    │   ├── ifc/              # IFC格式模型
    │   ├── obj/              # OBJ格式模型
    │   ├── pcd/              # PCD格式点云
    │   └── revit2020/        # Revit 2020格式模型
    └── test.py               # 点云测试脚本
```

## 软件架构

本项目采用模块化架构设计，主要包括以下几个核心模块：

1. **模型转换模块**：

   - 实现Revit模型到点云数据的转换流程
   - 支持IFC、OBJ等中间格式
   - 确保模型材质和几何信息的准确转换
2. **点云处理模块**：

   - 点云数据的加载、保存和格式转换
   - 点云滤波、下采样等预处理操作
   - 点云可视化和信息提取
3. **质量检测模块**：

   - 预制构件尺寸偏差检测
   - 表面缺陷识别
   - 安装精度评估
4. **结果可视化模块**：

   - 检测结果的三维可视化
   - 质量评估报告生成

## 安装教程

### 系统要求

- Windows 10/11 64位操作系统
- Python 3.12及以上版本
- Revit 2020（用于模型转换）
- CloudCompare（用于点云处理和可视化）

### 环境配置

1. **安装Python依赖**：

   ```bash
   pip install -r requirements.txt
   ```

2. **安装辅助软件**：

   - Revit 2020：用于BIM模型的创建和导出
   - CloudCompare：用于点云数据的处理和可视化
3. **下载测试资源**：
   从百度网盘下载测试资源文件：

   - 链接: [https://pan.baidu.com/s/1IlWkHDnlBPdxBeajxfq-WQ?pwd=e17n](https://pan.baidu.com/s/1IlWkHDnlBPdxBeajxfq-WQ?pwd=e17n)
   - 提取码: e17n
   - 将下载的资源解压到 `mock/res/` 目录下

## 使用说明

### Revit模型转点云

参考 `mock/Revit2Point.md` 文档，完成Revit模型到点云数据的转换：

1. 打开Revit模型并导出为IFC格式
2. 将IFC文件转换为OBJ格式（保留材质信息）
3. 使用CloudCompare将OBJ模型转换为PCD点云格式

### 点云测试

运行 `mock/test.py` 脚本测试点云数据：

```bash
python mock/test.py
```

该脚本将：

- 加载指定的点云文件
- 检查文件是否存在和点云是否为空
- 打印点云的基本信息（点数量、边界框等）
- 显示点云的可视化效果

## 开发计划

- [X] 完成点云数据处理模块的开发
- [ ] 实现预制构件尺寸偏差检测算法
- [ ] 开发表面缺陷识别功能
- [ ] 设计质量评估报告生成模块
- [ ] 开发用户界面

## 技术栈

- **编程语言**：Python 3.12+
- **点云处理**：Open3D
- **BIM模型**：Revit 2020、IFC格式
- **点云可视化**：CloudCompare

## 许可证

本项目为内部项目，暂无开源计划，仅内部使用。

## 联系方式

如有问题或建议，请联系项目负责人：

- 邮箱：<xiaopenggu@qq.com>
- 电话：15827027507

---

**备注**：本项目目前处于开发阶段，部分功能可能尚未实现或处于测试阶段。
