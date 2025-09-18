# FEA 有限元分析框架架构文档

## 概述

FEA是一个基于PyTorch的有限元分析框架，支持非线性材料、接触分析等高级功能。该框架采用面向对象的设计，具有清晰的层次结构和模块化的组织方式。

## 整体架构

```
FEA/
├── FEA/                    # 核心模块
│   ├── __init__.py         # 模块入口，提供from_inp()等顶层接口
│   ├── controller.py       # FEAController主控制器
│   ├── inp.py             # INP文件解析器
│   ├── assemble/          # 装配模块 (核心组织层)
│   │   ├── assembly.py     # Assembly主装配类
│   │   ├── part.py         # Part和Instance
│   │   ├── reference_points.py # 参考点
│   │   ├── elements/       # 单元模块
│   │   │   ├── base.py     # 单元基类
│   │   │   ├── C3/         # 3D单元
│   │   │   └── materials/  # 材料模型
│   │   ├── loads/          # 载荷模块
│   │   │   ├── base.py     # 载荷基类
│   │   │   ├── contact.py  # 接触载荷
│   │   │   └── ...         # 其他载荷类型
│   │   └── constraints/    # 约束模块
│   │       ├── base.py     # 约束基类
│   │       └── ...         # 具体约束类型
│   └── solver/            # 求解器模块
├── ui/                    # 用户界面模块
├── tests/                 # 测试用例
└── Docs/                  # 文档
```

## 核心组件架构

### 1. 控制层 (Controller Layer)

#### FEAController
- **文件位置**: `FEA/controller.py`
- **功能**: 主控制器，协调装配(Assembly)和求解器(Solver)
- **主要方法**:
  - `initialize()`: 初始化模型
  - `solve()`: 执行求解过程
- **属性**:
  - `assembly`: 装配对象
  - `solver`: 求解器对象

### 2. 装配层 (Assembly Layer)

装配层是FEA框架的核心组织层，统一管理所有有限元模型组件，包括几何、材料、载荷、约束等。

#### Assembly (装配)
- **文件位置**: `FEA/assemble/assembly.py`
- **功能**: 统一管理整个有限元模型的所有组件
- **核心属性**:
  - `_parts`: 部件字典
  - `_instances`: 实例字典
  - `_surfaces`: 表面集合
  - `_reference_points`: 参考点
  - `_loads`: 载荷集合
  - `_constraints`: 约束集合
  - `RGC`: 冗余广义坐标
  - `GC`: 广义坐标

#### 装配层子模块

##### 几何组件
- **Part (部件)**: 定义几何部件，包含节点、单元、材料等信息
- **Instance (实例)**: 部件的实例化，可进行变换和定位
- **ReferencePoint (参考点)**: 定义参考点，用于约束和载荷的施加

##### 单元模块 (`elements/`)
- **BaseElement**: 单元基类
- **C3D系列**: 3D单元 (C3D4, C3D8, C3D10, C3D20等)
- **材料模块**: 线弹性、超弹性等材料模型
- **表面单元**: T3, T6, Q4, Q8等表面单元

##### 载荷模块 (`loads/`)
- **BaseLoad**: 载荷基类
- **Concentrate_Force**: 集中力载荷
- **Pressure**: 表面压力载荷
- **Contact**: 接触载荷
- **BodyForce**: 体力载荷

##### 约束模块 (`constraints/`)
- **BaseConstraint**: 约束基类
- **Boundary_Condition**: 位移边界条件
- **Couple**: 耦合约束

### 3. 求解器层 (Solver Layer)

```
solver/
├── __init__.py
├── basesolver.py         # BaseSolver基类
├── static_implicit.py    # 静力隐式求解器
└── linear_solver.py      # 线性求解器
```

#### 求解器类型
- **StaticImplicitSolver**: 静力隐式求解器
- **LinearSolver**: 线性求解器

## 数据流架构

### 1. 模型构建流程
```
INP文件 → FEA_INP解析 → from_inp() → FEAController
                                       ↓
                                   Assembly (装配层)
                                       ↓
                          ┌─────────────┼─────────────┐
                          ↓             ↓             ↓
                      几何组件        载荷组件        约束组件
                    (Parts/Instances) (Loads)      (Constraints)
                          ↓
                    Elements + Materials
```

### 2. 求解流程
```
FEAController.initialize() → Assembly.initialize() → 各组件初始化
                                 ↓                     ↓
                         RGC空间分配              坐标系统构建
                                 ↓
                     Solver.initialize() → 求解器配置
                                 ↓
FEAController.solve() → Solver.solve() → Assembly刚度矩阵装配 → 结果输出
```

## 坐标系统

### 广义坐标系统 (GC/RGC)

#### RGC申请与分配机制
系统采用动态的自由度申请机制，各个组件根据自身需求申请RGC空间：

- **实例 (Instance)**: 申请节点数量 × 3个平移自由度的RGC空间
  - `_RGC_requirements = self.part.nodes.shape` (节点数 × 3)
  
- **参考点 (ReferencePoint)**: 申请6个自由度 (3个平移 + 3个旋转)
  - `_RGC_requirements = 6`
  
- **载荷 (Loads)**: 根据载荷类型申请相应的RGC空间
  - 基础载荷：继承自`BaseLoad`，可申请额外的内部变量空间
  - 接触载荷：可能需要额外的拉格朗日乘子空间
  
- **约束 (Constraints)**: 根据约束类型申请RGC空间
  - 边界条件：不申请额外空间，而是修改现有RGC的自由度状态
  - 耦合约束：可能申请拉格朗日乘子空间

#### RGC分配流程
1. **分配阶段** (`Assembly.initialize()`):
   ```python
   # 为每个组件分配RGC空间
   for ins in self._instances.keys():
       RGC_index = self._allocate_RGC(size=self._instances[ins]._RGC_requirements)
   
   for rp in self._reference_points.keys():
       RGC_index = self._allocate_RGC(size=self._reference_points[rp]._RGC_requirements)
   
   for load in self._loads.keys():
       RGC_index = self._allocate_RGC(size=self._loads[load]._RGC_requirements)
   ```

2. **约束处理**: 通过`set_required_DoFs()`方法标记哪些自由度被约束
3. **RGC到GC映射**: 只有未被约束的自由度参与求解
   - `GC = RGC[remain_index]` (提取活跃自由度)
   - `RGC = apply_constraints(GC)` (应用约束条件)

#### 坐标系统特点
- **动态分配**: 根据模型复杂度动态分配内存
- **分层管理**: 每个组件管理自己的RGC段
- **约束解耦**: 约束通过索引映射实现，不改变RGC结构
- **GPU友好**: 基于PyTorch张量，支持GPU加速计算

## 扩展性设计

### 1. 插件化架构
- 所有组件都基于基类设计，支持继承扩展
- 单元、材料、载荷、约束都可以通过继承基类来添加新类型

### 2. 设备无关性
- 基于PyTorch，支持CPU/GPU计算
- 通过`device`属性统一管理计算设备

### 3. 模块化设计
- 每个功能模块独立，便于维护和测试
- 清晰的接口定义，便于组合使用

## 使用示例

### 基本使用流程
```python
import FEA

# 从INP文件加载模型
inp = FEA.FEA_INP()
inp.read_inp('model.inp')
controller = FEA.from_inp(inp)

# 设置求解器
controller.solver = FEA.solver.StaticImplicitSolver()

# 初始化和求解
controller.initialize()
result = controller.solve()
```

## 特色功能

### 1. 自动微分
- 基于PyTorch的自动微分功能
- 支持梯度计算和优化

### 2. 高阶单元
- 支持二次单元 (C3D10, C3D15, C3D20等)
- 自动生成二次单元的功能

### 3. 接触分析
- 支持自接触和多体接触
- 接触力的自动计算

### 4. 表面处理
- 丰富的表面单元类型
- 表面集合的管理和操作

## 开发指南

### 添加新单元类型
1. 继承`BaseElement`或相应的基类
2. 实现必要的方法 (形函数、雅可比等)
3. 在`__init__.py`中注册新单元

### 添加新材料模型
1. 继承`Materials_Base`
2. 实现应力-应变关系
3. 在材料初始化函数中添加分支

### 添加新载荷类型
1. 继承`BaseLoad`
2. 实现载荷计算方法
3. 在载荷模块中注册

## 性能优化

### 1. 内存管理
- 使用PyTorch张量进行高效内存管理
- 避免不必要的数据复制

### 2. 并行计算
- 支持GPU加速计算
- 向量化运算优化

### 3. 稀疏矩阵
- 使用稀疏矩阵存储刚度矩阵
- 高效的线性代数运算

---

*此文档描述了FEA有限元分析框架的整体架构和设计理念，为开发者提供了全面的结构化认识。*
