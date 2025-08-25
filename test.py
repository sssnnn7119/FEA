import torch

# 1. 创建示例数据: 10000个样本，每个样本3个输入
batch_size = 10000
inputs = torch.randn(batch_size, 3, requires_grad=True)  # 形状: [10000, 3]

# 2. 定义计算函数 (与一阶导数示例相同)
outputs = inputs[:, 0]**2 + inputs[:, 1] * inputs[:, 2] + torch.sin(inputs[:, 2])
# outputs形状: [10000]

# 3. 先计算一阶导数
grads_first = torch.autograd.grad(
    outputs=outputs,
    inputs=inputs,
    grad_outputs=torch.ones_like(outputs),
    create_graph=True,  # 需要创建计算图以进行二阶导数计算
    retain_graph=True,
    only_inputs=True
)[0]  # 形状: [10000, 3]

# 4. 计算二阶导数 (Hessian矩阵)
# 为每个输入维度单独计算梯度，然后组合成Hessian矩阵
hessians = []
for i in range(inputs.size(1)):  # 对每个输入维度循环
    # 计算一阶导数对第i个输入的梯度
    grad_second = torch.autograd.grad(
        outputs=grads_first[:, i],
        inputs=inputs,
        grad_outputs=torch.ones_like(grads_first[:, i]),
        create_graph=False,
        retain_graph=True,
        only_inputs=True
    )[0]  # 形状: [10000, 3]
    hessians.append(grad_second.unsqueeze(1))  # 增加一个维度以便拼接

# 拼接得到Hessian矩阵，形状变为[10000, 3, 3]
hessians = torch.cat(hessians, dim=1)

# 5. 查看结果
print(f"Hessian矩阵形状: {hessians.shape}")  # 应该是 [10000, 3, 3]
print(f"第一个样本的Hessian矩阵:\n{hessians[0]}")

# 验证（可选）：手动计算第一个样本的二阶导数
x = inputs[0].detach().clone()
x.requires_grad_(True)
y = x[0]**2 + x[1] * x[2] + torch.sin(x[2])

# 一阶导数
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

# 二阶导数
hessian_manual = []
for i in range(3):
    hessian_row = torch.autograd.grad(dy_dx[i], x, retain_graph=True)[0]
    hessian_manual.append(hessian_row)
hessian_manual = torch.stack(hessian_manual)

print(f"第一个样本的手动计算Hessian矩阵:\n{hessian_manual}")
    