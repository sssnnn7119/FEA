# Sensitivity Analysis (Adjoint Method)

Adjoint sensitivity analysis 是一种用于计算目标函数对设计变量的梯度的有效方法，常在优化和机器学习任务（如拓扑优化或基于物理知识的神经网络训练）中使用。在 \`torchfea\` 中，我们可以使用伴随法（Adjoint method）高效地计算敏度（梯度）。

## 基本用法

在 \`torchfea\` 的 `StaticImplicitSolver` 中提供了 \`get_sensitivity\` 方法，用于自动管理伴随方程的求解和梯度的自动求导计算。

### 1. 导入库与模型求解

首先，设置模型并求解一次静力学结果。在运行敏度分析之前，你需要一个收敛的静力学解（\`fe_result\`）。以下示例代码演示了该过程：

```python
import torch
import torchfea

# 初始化模型解析参数
fem = torchfea.FEA_INP()
fem.read_inp('example.inp')
fe = torchfea.from_inp(fem)

# 分配求解器
fe.solver = torchfea.solver.StaticImplicitSolver()

# 添加载荷和边界条件
fe.assembly.add_load(...)
fe.assembly.add_boundary(...)

# 初始化并求解模型
fe.initialize()
feresult = fe.solve(tol_error=1e-6)
```

### 2. 定义设计变量的应用函数 (`apply_func`)

敏度分析需要知道设计变量如何影响有限元系统。你需要定义一个回调函数 `apply_func`，该函数会在敏度计算中修改 `Assembly` 的属性。这些修改**必须能通过 PyTorch 追踪其计算图 (Autograd)**。
例如，以节点坐标作为设计变量：

```python
def apply_design_vars(assembly: torchfea.Assembly, design_vars: torch.Tensor) -> None:
    # 获取需要修改的部件
    part = assembly.get_part('final_model')
    # 使用传入的含有梯度信息的 design_vars 更新模型属性
    # 注意 reshape 需要匹配原来的形状，避免破坏计算图
    part.nodes = design_vars.reshape(part.nodes.shape)
```

### 3. 定义目标函数 (`compute_objective_func`)

接下来，定义一个回调函数用来计算优化目标。函数的参数是节点的广义坐标（`GC`，包含了位移信息）以及当前的有限元装配体（`assembly`）。该函数应返回一个标量 `Tensor`。
例如，目标函数是针对某个特定自由度（如末端参考点位移）的最小化：

```python
def compute_objective(GC: torch.Tensor, assembly: torchfea.Assembly) -> torch.Tensor:
    # 假设需要最小化索引为 -2 的自由度的值
    obj = GC[-2]
    return obj
```

### 4. 计算敏感度梯度

通过调用求解器的 `get_sensitivity` 函数，将之前得到的结果和回调函数输入，即可完成梯度的自动计算：

```python
# 获取用于求导的设计变量的初始值 (必须要求是一维/多维的 torch.Tensor)
part = fe.assembly.get_part('final_model')
initial_design_vars = part.nodes.clone().detach().reshape(-1)

# 调用伴随敏度推导
grad_sensi = fe.solver.get_sensitivity(
    fe_result=feresult,
    design_vars=initial_design_vars,
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective,
)

print('敏感度梯度:', grad_sensi)
```

## 计算机制解释

`get_sensitivity` 采用计算图追踪与伴随状态方程（Adjoint Equation）相结合的离散伴随法：

1. **设置状态量**：利用 `apply_func` 将设计变量附着到了模型中，然后重建有限元系统的计算图对象。
2. **伴随载荷构建**：对 `compute_objective_func` 进行一次前向反向计算来得到目标对位移的偏导数 $\frac{\partial Obj}{\partial \boldsymbol{U}}$。
3. **伴随变量求解（Adjoint Equation Solve）**：使用求解静力学平衡方程时产生的刚度矩阵（已分解完成，使得求解非常快），求解伴随系统方程 $\boldsymbol{K} \boldsymbol{\lambda} = - \frac{\partial Obj}{\partial \boldsymbol{U}}$。
4. **梯度重构（Total Sensitivity）**：通过求内积重构残差功方程并调用 PyTorch 自动微分（`work.backward()`），最终求出目标对象对全部设计变量的总导数 $\frac{\partial Obj}{\partial X}$。

使用 \`torchfea\` 中的这种设计方式，保证了复杂非线性边界（如接触非线性）、材料非线性、大变形场景下的精确梯度计算，并且整个过程只需求解一次额外的线性方程组，相比由于摄动法求梯度的计算量呈指数级下降。

## 雅可比灵敏度分析 (Jacobian Sensitivity)

在进行高级分析和设计优化时（如针对载荷参数不确定性的鲁棒性优化、追踪特定自由度的荷载位移曲线斜率等），有时并不只是关注单纯的位移，还会高度关注系统的响应对特定载荷参数的变化率（即雅可比矩阵，Jacobian）。

如果目标函数不仅依赖于系统状态变量（自由度 $u_i$），还依赖于系统位移对载荷参数的雅可比矩阵（$\frac{\partial u_i}{\partial p_n}$），可以使用 `StaticImplicitSolver` 提供的 `get_jacobian_sensitivity` 方法。该方法通过多重伴随系统，同时处理刚度矩阵和残余力的自动微分解析，实现了高阶（包含雅可比项）敏感度的高效求解。

### 用法举例

与基础的伴随敏度分析类似，使用雅可比敏度分析同样需要准备好收敛的静力学解 `fe_result` 和设计变量。主要的区别在于：你需要指定关注哪些影响雅可比的载荷参数，且目标函数的参数输入有所不同。

#### 1. 定义受雅可比影响的目标函数

在使用该功能时，`compute_objective_func` 除了接受全量自由度张量 `GC` (对应 $u$) 和 `assembly` 外，还会接受 `jacobian` $\frac{\partial u_i}{\partial p_n}$ 张量，以及 `other_args`。

```python
def compute_jacobian_objective(
    GC: torch.Tensor, 
    jacobian: torch.Tensor, 
    assembly: torchfea.Assembly,
    other_args: dict = None
) -> torch.Tensor:
    # 例如：希望优化模型，使得某个特定自由度（索引-2）对外界第一个载荷参数变化最不敏感（鲁棒性最强）
    # jacobian 是列拼合后的总雅可比矩阵张量，shape: [num_dofs, num_load_params]
    obj = (jacobian[-2, 0] ** 2)
    return obj
```

#### 2. 调用求解器推导雅可比敏度

提供需要计算偏导的载荷名称列表（`load_names`），并传入目标函数，会自动推导目标函数对设计变量的偏导数：

```python
# 指定需要求雅可比相关的载荷名称，例如面载荷 'pressure_1'
focus_loads = ['pressure_1']

# 调用雅可比敏度分析
jacobian_outputs, jacobian_sensi = fe.solver.get_jacobian_sensitivity(
    fe_result=feresult,
    design_vars=initial_design_vars,
    load_names=focus_loads,
    apply_func=apply_design_vars,
    compute_objective_func=compute_jacobian_objective,
    other_args=None  # 可附加任何其它 tensor 传给目标函数
)

print('计算获得的各个载荷雅可比字典:', jacobian_outputs)
print('目标函数的雅可比设计敏感度梯度:', jacobian_sensi)
```

--

## 理论与计算机制 (Theory & Implementation)

### 常用符号表

| 符号 | 物理及数学意义 |
| :---: | :--- |
| $L$ | 优化目标函数 (Objective function) |
| $u_i$ | 节点的广义自由度或系统位移 (Generalized DoFs) |
| $b_m$ | 用于优化的设计变量 (Design variables) |
| $p_n$ | 外部特定的系统参量，如载荷大小 (External parameters) |
| $R_i$ | 系统残余力 (Residual force, Internal force - External force) |
| $K_{ik}$ | 对称系统的切线刚度矩阵 (Tangent stiffness matrix) |
| $J_{in}$ | 系统位移对待定参数的雅可比矩阵响应 (Jacobian matrix) |
| $\lambda_i$ | 计算伴随方程引入的一阶伴随变量 (Primary adjoint variables) |
| $\lambda_{in}^*, \lambda_i^{**}$ | 计算雅可比伴随方程引入的二次伴随变量 (Secondary adjoint variables) |

### 1. 位移场灵敏度推导

我们首先定义位移场相关目标函数

$$
L = L(u_i, b_m)
$$

在基础分析中，静力平衡方程由残余力为零表示：

$$
R_i(u_k, b_m) = 0
$$

利用链式法则求其对设计变量 $b_m$ 的全导数为：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \frac{\partial L}{\partial u_k} \frac{\mathrm{d} u_k}{\mathrm{d} b_m}
$$

上式中隐式的 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m}$（位移对设计变量的偏导）直接计算代价极高。我们对平衡方程 $R_i = 0$ 再次对 $b_m$ 求全导数以建立联系：

$$
K_{ik} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} + \frac{\partial R_i}{\partial b_m} = 0 \quad \Rightarrow \quad \frac{\mathrm{d} u_k}{\mathrm{d} b_m} = - K^{-1}_{kr} \frac{\partial R_r}{\partial b_m}
$$

将 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m}$ 代入总灵敏度方程中，即得：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} - \frac{\partial L}{\partial u_k} K^{-1}_{kr} \frac{\partial R_r}{\partial b_m}
$$

对于复杂的系统，我们不想分别求每一个设计变量的导数。此时，我们定义伴随变量 $\lambda_r$ 满足伴随方程：

$$
\lambda_r = - K^{-1}_{rk} \frac{\partial L}{\partial u_k}
$$

可将复杂项化简。最终的总灵敏度退化为计算如下内积：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \lambda_r \frac{\partial R_r}{\partial b_m}
$$

最中对上述表达式进行自动微分即可得到目标函数对设计变量的总导数。

### 2. 包含雅可比的高阶敏度推导

在更复杂的优化场景中（例如针对参数不确定性的鲁棒性优化，或多载荷工况下的灵敏度分析），目标函数可能不仅依赖于系统当前的位移 $u_i$，还会显式依赖于系统位移对外设参数 $p_n$（如特定载荷边界大小、材料参数等）的雅可比矩阵 $J_{in} = \frac{\partial u_i}{\partial p_n}$。此时，目标函数可以一般化地表示为：

$$
L = L(u_i, J_{in}, b_m)
$$

对于给定的任意系统参数 $p_n$，系统必须依然满足静力等效平衡方程：

$$
R_j(u_i, b_m, p_n) = 0
$$

利用隐函数定理直接对上式关于外部参数 $p_n$ 求解全导数，即可得到系统雅可比矩阵的控制方程：

$$
K_{ji} \frac{\partial u_i}{\partial p_n} + \frac{\partial R_j}{\partial p_n} = 0 \quad \Rightarrow \quad J_{in} = - K_{ij}^{-1} \frac{\partial R_j}{\partial p_n}
$$

在此基础上，由于目标函数引入了雅可比依赖，根据多元微积分的链式法则，目标函数对设计变量 $b_m$ 的全导数扩展为：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \frac{\partial L}{\partial u_k} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} + \frac{\partial L}{\partial J_{in}} \frac{\mathrm{d} J_{in}}{\mathrm{d} b_m}
$$

推导的难点集中于上式的第三项 $\frac{\mathrm{d} J_{in}}{\mathrm{d} b_m}$。为了求解这一项，我们将此前得到的雅可比平衡等式 $\left(K_{ji} J_{in} + \frac{\partial R_j}{\partial p_n} = 0\right)$ 对设计变量 $b_m$ 此度求取全导数。需注意的是，刚度矩阵 $K$ 和残余力 $R$ 均包含对位移 $u$ 和设计变量 $b$ 的隐式依赖，我们必须严格应用链式扩展：

$$
K_{ji} \frac{\mathrm{d} J_{in}}{\mathrm{d} b_m} + \left( \frac{\partial K_{ji}}{\partial b_m} + \frac{\partial K_{ji}}{\partial u_k} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} \right) J_{in} + \left( \frac{\partial^2 R_j}{\partial b_m \partial p_n} + \frac{\partial^2 R_j}{\partial u_k \partial p_n} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} \right) = 0
$$

为揭示内部机制，我们将上式中含有 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m}$ 的项提取得出，并在等式两端移项整理：

$$
K_{ji} \frac{\mathrm{d} J_{in}}{\mathrm{d} b_m} = - \left( \frac{\partial K_{ji}}{\partial u_k} J_{in} + \frac{\partial^2 R_j}{\partial u_k \partial p_n} \right) \frac{\mathrm{d} u_k}{\mathrm{d} b_m} - \left( \frac{\partial K_{ji}}{\partial b_m} J_{in} + \frac{\partial^2 R_j}{\partial b_m \partial p_n} \right)
$$

观察上述等式右侧的系数，括号内的物理意义分别代表对应参数关于 $p_n$ 的全微分。结合材料线弹性或严格切线刚度特性 $\frac{\partial K_{ji}}{\partial u_k} = \frac{\partial K_{jk}}{\partial u_i}$，我们引入全微分算子 $\frac{\mathbf{d}}{\mathbf{d} p_n}$ 进行聚合：

$$
K_{ji} \frac{\mathrm{d} J_{in}}{\mathrm{d} b_m} = - \frac{\mathbf{d} K_{jk}}{\mathbf{d} p_n} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} - \frac{\partial}{\partial b_m}\left( \frac{\mathbf{d} R_j}{\mathbf{d} p_n} \right) 
$$

回忆第一节中的位移场灵敏度基础结论 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m} = - K_{kr}^{-1} \frac{\partial R_r}{\partial b_m}$。将其代入上式，并同时在方程两侧左乘逆刚度矩阵 $K^{-1}$ 进行指标置换转化，即得雅可比全导数的显式代数表达式：

$$
\frac{\mathrm{d} J_{jn}}{\mathrm{d} b_m} = K^{-1}_{jr} \left[\frac{\mathbf{d} K_{rs}}{\mathbf{d} p_n} K^{-1}_{si} \frac{\partial R_i}{\partial b_m} - \frac{\partial}{\partial b_m} \left( \frac{\mathbf{d}R_r}{\mathbf{d}p_n} \right)\right]
$$

带入总灵敏度表达式中的雅可比项:

$$
\frac{\partial L}{\partial J_{jn}} \frac{\mathrm{d} J_{jn}}{\mathrm{d} b_m} =\frac{\partial L}{\partial J_{jn}} K^{-1}_{jr} \left[\frac{\mathbf{d} K_{rs}}{\mathbf{d} p_n} K^{-1}_{si} \frac{\partial R_i}{\partial b_m}- \frac{\partial}{\partial b_m} \left( \frac{\mathbf{d}R_r}{\mathbf{d}p_n} \right)\right]
$$

为了避免显式计算和存储极度庞大的 $\mathbf{d} K_{rs}/\mathbf{d} p_n$ 项与雅可比偏微分耦合张量，在代码中 `torchfea` 巧妙运用了高阶伴随方法（Secondary Adjoint Method），只需要执行**两次**额外的线性方程组顺次求解，即可计算上式所描述的总灵敏度修正项：

1. **一阶伴随方程求解**：引入雅可比一阶伴随变量 $\lambda_{rn}^*$，通过求解等价线性系统得到：
   $$
   K_{rj} \lambda_{rn}^* = - \frac{\partial L}{\partial J_{jn}}
   $$

2. **二阶伴随方程求解**：将得到的一阶伴随变量 $\lambda_{rn}^*$ 继续应用于刚度矩阵偏导投影，引入二阶伴随变量 $\lambda_i^{**}$，其对应的线性系统为：
   $$
   K_{is} \lambda_i^{**} = \frac{\mathbf{d} K_{rs}}{\mathbf{d} p_n} \lambda_{rn}^*
   $$

由于以上两步均可以直接复用原位移场静力分析已经完成因式分解（$LDL^T$ 分解）的 $K$ 矩阵，这些额外求解步骤的计算成本微乎其微。

最终组合位移场相关的伴随敏度项，系统对设计变量 $b_m$ 的总灵敏度可以通过自动微分自动拼装，完整表达式收敛为：

$$
\frac{\mathrm{d}L}{\mathrm{d}b_m} = 
\frac{\partial L}{\partial b_m} + \lambda_r \frac{\partial R_r}{\partial b_m} + \lambda_i^{**} \frac{\partial R_i}{\partial b_m} + \lambda_{rn}^{*} \frac{\partial}{\partial b_m} \left( \frac{\mathbf{d}R_r}{\mathbf{d}p_n} \right)
$$