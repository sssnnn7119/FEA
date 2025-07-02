# FEA 项目文件结构

## 项目根目录

```
FEA/
├── README.md                   # 项目说明文档
├── requirements.txt            # Python依赖包列表
├── setup.py                   # 项目安装配置文件
├── .gitignore                 # Git忽略文件配置
└── LICENSE                    # 项目许可证
```

## 核心FEA模块

```
FEA/
├── __init__.py                # FEA包初始化，导入主要类和函数
├── Main.py                    # FEA主类，包含求解器和模型管理
├── Main copy.py               # FEA主类的备份版本
├── FEA_INP.py                # INP文件读取和解析模块
├── obj_base.py               # FEA对象基类定义
└── reference_points.py       # 参考点类定义
```

## 有限元单元模块

```
elements/
├── __init__.py               # 单元模块初始化
├── base.py                   # 单元基类定义
├── materials.py              # 材料模型定义和初始化
├── C3/                       # 三维单元
│   ├── __init__.py
│   ├── C3base.py             # 三维单元基类
│   ├── C3D4.py               # 4节点四面体单元
│   ├── C3D6.py               # 6节点楔形单元
│   ├── C3D8.py               # 8节点六面体单元
│   ├── C3D8R.py              # 8节点简化积分六面体单元
│   ├── C3D10.py              # 10节点二次四面体单元
│   ├── C3D15.py              # 15节点二次楔形单元
│   └── C3D20.py              # 20节点二次六面体单元
└── S3/                       # 壳单元
    ├── __init__.py
    └── S3.py                 # 3节点三角形壳单元
```

## 载荷模块

```
loads/
├── __init__.py               # 载荷模块初始化
├── base.py                   # 载荷基类定义
├── concentrate_force.py      # 集中力载荷
├── pressure.py               # 压力载荷
└── moment.py                 # 力矩载荷
```

## 约束模块

```
constraints/
├── __init__.py               # 约束模块初始化
├── base.py                   # 约束基类定义
├── boundary_condition.py     # 边界条件约束
├── coupling.py               # 耦合约束
```

## 求解器模块

```
solver/
├── __init__.py               # 求解器模块初始化
├── linear_solver.py          # 线性求解器
└── lbfgs.py                  # L-BFGS优化求解器
```

## 用户界面模块

```
ui/
├── __init__.py               # UI模块初始化
├── main_window.py            # 主窗口界面
├── fea_widget.py             # FEA操作主界面
├── object_manager.py         # 对象管理器界面
├── visualization.py          # 可视化模块
└── dialogs/                  # 对话框模块
    ├── __init__.py
    ├── boundary_condition_dialog.py    # 边界条件设置对话框
    ├── load_dialog.py                  # 载荷设置对话框
    ├── reference_point_dialog.py       # 参考点设置对话框
    └── coupling_dialog.py              # 耦合约束设置对话框
```

## 文档目录

```
Docs/
├── structure.md              # 项目结构说明（当前文件）
├── api/                      # API文档
│   ├── main.md               # 主模块API
│   ├── elements.md           # 单元模块API
│   ├── loads.md              # 载荷模块API
│   └── constraints.md        # 约束模块API
├── tutorials/                # 教程文档
│   ├── getting_started.md    # 入门指南
│   ├── basic_analysis.md     # 基础分析教程
│   └── advanced_features.md  # 高级功能教程
└── examples/                 # 示例文档
    ├── beam_analysis.md      # 梁分析示例
    ├── plate_analysis.md     # 板分析示例
    └── thermal_analysis.md   # 热分析示例
```

## 测试目录

```
tests/
├── __init__.py               # 测试包初始化
├── test_main.py              # 主模块测试
├── test_elements.py          # 单元模块测试
├── test_loads.py             # 载荷模块测试
├── test_constraints.py       # 约束模块测试
├── test_ui.py                # 界面模块测试
└── fixtures/                 # 测试数据
    ├── meshes/               # 测试网格文件
    ├── materials/            # 测试材料数据
    └── results/              # 参考结果
```

## 示例目录

```
examples/
├── basic/                    # 基础示例
│   ├── simple_beam.py        # 简单梁分析
│   ├── cantilever.py         # 悬臂梁分析
│   └── truss.py              # 桁架分析
├── advanced/                 # 高级示例
│   ├── nonlinear.py          # 非线性分析
│   ├── contact.py            # 接触分析
│   └── optimization.py      # 优化分析
└── data/                     # 示例数据文件
    ├── meshes/               # 网格文件
    ├── materials.json        # 材料属性文件
    └── loads.json            # 载荷定义文件
```

## 主要功能模块说明

### FEA核心模块

- **Main.py**: 包含FEA_Main主类，实现有限元求解的核心功能

  - 模型初始化和求解
  - 载荷和约束管理
  - 线性和非线性求解算法
  - 几何坐标转换（RGC/GC）
- **FEA_INP.py**: INP文件格式支持

  - 读取Abaqus INP文件
  - 解析部件、材料、单元信息
  - 提取节点集、单元集、表面集

### 单元模块

- **C3D系列**: 三维连续单元

  - C3D4/C3D10: 线性/二次四面体单元
  - C3D8/C3D20: 线性/二次六面体单元
  - C3D6/C3D15: 线性/二次楔形单元
  - 支持不同积分方式和单元阶数
- **材料模型**:

  - Neo-Hookean超弹性材料
  - 线弹性材料
  - 材料参数初始化和管理

### 载荷和约束

- **载荷类型**:

  - 集中力载荷
  - 压力载荷（表面载荷）
  - 力矩载荷
- **约束类型**:

  - 边界条件约束
  - 耦合约束
  - 刚体约束

### 用户界面

- **主界面**: 基于PyQt5的图形用户界面
- **对象管理器**: 管理边界条件、载荷、参考点等对象
- **可视化**: 基于Mayavi的3D可视化
- **材料属性设置**: 支持不同材料模型参数设置

### 求解器

- **线性求解器**: 支持共轭梯度法和直接法求解
- **非线性求解器**: Newton-Raphson方法
- **优化求解器**: L-BFGS方法

## 依赖库

- **PyTorch**: 张量计算和自动微分
- **NumPy**: 数值计算
- **PyQt5**: 图形用户界面
- **Mayavi**: 3D可视化
- **SciPy**: 科学计算

## 文件命名规范

- 类文件使用PascalCase命名
- 模块文件使用snake_case命名
- 配置文件使用小写字母和下划线
- 文档文件使用小写字母和下划线
