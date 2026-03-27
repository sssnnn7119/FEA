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

\`get_sensitivity\` 采用计算图追踪与伴随状态方程（Adjoint Equation Equation）相结合的离散伴随法：

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

### 1. 基础伴随敏感度推导

在基础分析中，静力平衡方程由残余力为零表示：
$$ R_i(u_k, b_m) = 0 $$
其中 $R$ 是残余力（Internal force - External force），$u$ 为广义自由度（GC），$b$ 为设计变量（Design variables）。

假设有一个目标函数 $L = L(u_i, b_m)$。我们要求其对设计变量 $b_m$ 的全导数：
$$ \frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \frac{\partial L}{\partial u_i} \frac{\partial u_i}{\partial b_m} $$

由于直接计算隐式的 $\frac{\partial u_i}{\partial b_m}$ 代价极高。我们引入伴随变量 $\lambda_i$，由于平衡方程恒为零 $R_i = 0$，可以得到恒等式：
$$ \lambda_i R_i(u_k, b_m) = 0 $$
对式子关于 $b_m$ 求全导数：
$$ \lambda_i \left( \frac{\partial R_i}{\partial b_m} + \frac{\partial R_i}{\partial u_k} \frac{\partial u_k}{\partial b_m} \right) = \lambda_i \left( \frac{\partial R_i}{\partial b_m} + K_{ik} \frac{\partial u_k}{\partial b_m} \right) = 0 $$
其中 $K_{ik} = \frac{\partial R_i}{\partial u_k}$ 为对称的切线刚度矩阵。

把上述绝对零项加到拉格朗日函数导数中：
$$ \frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \frac{\partial L}{\partial u_k} \frac{\partial u_k}{\partial b_m} + \lambda_i \left( \frac{\partial R_i}{\partial b_m} + K_{ik} \frac{\partial u_k}{\partial b_m} \right) $$
重排合并 $\frac{\partial u_k}{\partial b_m}$ 的系数：
$$ \frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \lambda_i \frac{\partial R_i}{\partial b_m} + \left( \frac{\partial L}{\partial u_k} + \lambda_i K_{ik} \right) \frac{\partial u_k}{\partial b_m} $$

我们将关于 $\frac{\partial u_k}{\partial b_m}$ 的大括号系数强行设为零（这就是**伴随方程**）：
$$ K_{ki} \lambda_i = - \frac{\partial L}{\partial u_k} $$
只要解出 $\lambda_i$（在代码中这是对应的第一伴随向量 $W_0$），上式末尾的复杂项就消除了。最终的总灵敏度退化为计算如下内积：
$$ \frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \lambda_i \frac{\partial R_i}{\partial b_m} $$
这就是 `torchfea` 基础灵敏度推导将一切难以求导的工作压成求解一次线性方程的技术原理。

### 2. 包含雅可比的高阶敏度推导

当目标函数不仅依赖于位移 $u_i$，还由于考虑鲁棒等高阶特性，依赖于位移对某种参数 $p_n$（例如外载荷参数）的雅可比矩阵 $y_{in} = \frac{\partial u_i}{\partial p_n}$ 时即：
$$ L = L(u_i, y_{in}, b_m) $$

对任意外部参数 $p_n$，系统总是维持静态平衡：
$$ R_j(u_i, b_m, p_n) = 0 $$
利用隐函数定理直接对其关于 $p_n$ 求全导：
$$ K_{ji} \frac{\partial u_i}{\partial p_n} + \frac{\partial R_j}{\partial p_n} = 0 \quad \Rightarrow \quad y_{in} = - K_{ij}^{-1} \frac{\partial R_j}{\partial p_n} $$

接着对考虑了雅可比参量的目标函数由链式法则全偏导法求灵敏度：
$$ \frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + \frac{\partial L}{\partial u_k} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} + \frac{\partial L}{\partial y_{in}} \frac{\mathrm{d} y_{in}}{\mathrm{d} b_m} $$

最棘手的是其中的第三部分 $\frac{\mathrm{d} y_{in}}{\mathrm{d} b_m}$。我们将前面的雅可比平衡等式 $\left(K_{ji} y_{in} + \frac{\partial R_j}{\partial p_n} = 0\right)$ 对设计变量 $b_m$ 再度求全导数，此时由于刚度矩阵 $K$ 是关于 $u_k$ 和 $b_m$ 的函数，必须应用链式法则：
$$ K_{ji} \frac{\mathrm{d} y_{in}}{\mathrm{d} b_m} + \left( \frac{\partial K_{ji}}{\partial b_m} + \frac{\partial K_{ji}}{\partial u_k} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} \right) y_{in} + \left( \frac{\partial^2 R_j}{\partial b_m \partial p_n} + \frac{\partial^2 R_j}{\partial u_k \partial p_n} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} \right) = 0 $$
合并含有 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m}$ 的项，并利用对称性 $\frac{\partial^2 R_j}{\partial u_k \partial p_n} + \frac{\partial K_{ji}}{\partial u_k} y_{in} = \frac{\mathrm{d} K_{jk}}{\mathrm{d} p_n}$，可以得到原来普通雅可比推导等式中被遗漏的核心项，补充归纳如下：
$$ K_{ji} \frac{\mathrm{d} y_{in}}{\mathrm{d} b_m} + \left( \frac{\partial K_{ji}}{\partial b_m} y_{in} + \frac{\partial^2 R_j}{\partial b_m \partial p_n} \right) + \frac{\mathrm{d} K_{jk}}{\mathrm{d} p_n} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} = 0 $$

整理即得：
$$ \frac{\mathrm{d} y_{in}}{\mathrm{d} b_m} = - K_{ji}^{-1} \left[ \left( \frac{\partial K_{js}}{\partial b_m} y_{sn} + \frac{\partial^2 R_j}{\partial b_m \partial p_n} \right) + \frac{\mathrm{d} K_{jk}}{\mathrm{d} p_n} \frac{\mathrm{d} u_k}{\mathrm{d} b_m} \right] $$

可见为了求解这一极其庞大的偏微分耦合，在代码中 `torchfea` 将隐式导数反代转为多次伴随方程处理：
1. 首先计算常规的第一伴随变量 $W_{0,i}$ 处理基态对 $u$ 的偏导：
$$ K_{ir} W_{0,r} = - \frac{\partial L}{\partial u_i} $$
2. 再为每一个雅可比相关的参量维 $n$ 构建第二级伴随变量 $W_{1, jn}$，用于处理 $\frac{\partial L}{\partial y_{in}}$ 与 $K^{-1}$ 的结合：
$$ K_{ji} W_{1, jn} = \frac{\partial L}{\partial y_{in}} $$
3. 由于方程中残余的 $\frac{\mathrm{d} u_k}{\mathrm{d} b_m} = - K_{kr}^{-1} \frac{\partial R_r}{\partial b_m}$，使得伴随推导中再生出**额外的刚度逆求解**，引出附加伴随变量（代码中转化为 `wKdpKinv` 对应的 $v_{rn}$）：
$$ K_{kr} v_{rn} = \frac{\mathrm{d} K_{jk}}{\mathrm{d} p_n} W_{1, jn} $$

把这些伴随解巧妙拼回整体导数中，经过一系列内积交换化简，所有难以显式求的项（如 $\frac{\partial K}{\partial b}$, $\frac{\partial y}{\partial b}$）最终都可以被转化为对 Autograd 前向-反向残余力计算图（jvp / jacobian 的巧妙分离）：
$$ \frac{\mathrm{d}L}{\mathrm{d}b_m} = \frac{\partial L}{\partial b_m} + W_{0, r} \frac{\partial R_r}{\partial b_m} + v_{rn} \frac{\partial R_r}{\partial b_m} - W_{1, jn} \left( \frac{\partial K_{js}}{\partial b_m} y_{sn} + \frac{\partial^2 R_j}{\partial b_m \partial p_n} \right) $$

这把 $O(M^2N)$ 的海森阵复杂性直接降为 $O(1)$ 的若干次线性系统求解，合并后针对雅可比相关的参量本质上**形成了二次独立的线性方程组求逆**（解 $W_1$ 与 $v$）。这既保证了严谨的全导数耦合包含效应，也维持了极小的显存开支。这就是 `get_jacobian_sensitivity` 背后的理论精髓。