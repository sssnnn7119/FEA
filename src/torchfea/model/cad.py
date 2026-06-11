from .part import Part
from .elements import initialize_element
import torch



def create_box(xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float, nx: int, ny: int, nz: int, element_name: str = 'C3D8'):
    """
    Create a box geometry defined by the given dimensions and number of elements.

    Args:
        xmin (float): Minimum x-coordinate of the box.
        xmax (float): Maximum x-coordinate of the box.
        ymin (float): Minimum y-coordinate of the box.
        ymax (float): Maximum y-coordinate of the box.
        zmin (float): Minimum z-coordinate of the box.
        zmax (float): Maximum z-coordinate of the box.
        nx (int): Number of elements along the x-axis.
        ny (int): Number of elements along the y-axis.
        nz (int): Number of elements along the z-axis.

    Returns:
        nodes (torch.Tensor): Tensor of shape (num_nodes, 3) containing the coordinates            of the nodes.
        elements (torch.Tensor): Tensor of shape (num_elements, 8) containing the node            indices for each element.
    """

    assert xmin < xmax, "xmin must be less than xmax"
    assert ymin < ymax, "ymin must be less than ymax"
    assert zmin < zmax, "zmin must be less than zmax"
    assert nx > 0, "nx must be greater than 0"
    assert ny > 0, "ny must be greater than 0"
    assert nz > 0, "nz must be greater than 0"

    x = torch.linspace(xmin, xmax, nx)
    y = torch.linspace(ymin, ymax, ny)
    z = torch.linspace(zmin, zmax, nz)

    nodes = torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3)

    # 步长
    stride_x = ny * nz
    stride_y = nz
    stride_z = 1
    
    # 创建单元（按 Abaqus 顺序）
    elements = torch.zeros(nx-1, ny-1, nz-1, 8, dtype=torch.long)
    
    # 基准节点（局部节点0，对应 Abaqus 节点1的位置）
    base = torch.arange(0, (nx-1) * stride_x, stride_x, dtype=torch.long)[:, None, None] \
        + torch.arange(0, (ny-1) * stride_y, stride_y, dtype=torch.long)[None, :, None] \
        + torch.arange(0, nz-1, dtype=torch.long)[None, None, :]
    
    # Abaqus 节点顺序映射
    # Abaqus 顺序: 1,2,3,4,5,6,7,8
    # 对应原顺序的偏移量
    elements[:, :, :, 0] = base                    # 节点1: 底面左下前
    elements[:, :, :, 1] = base + stride_x         # 节点2: 底面右下前
    elements[:, :, :, 2] = base + stride_x + stride_y  # 节点3: 底面右后前
    elements[:, :, :, 3] = base + stride_y         # 节点4: 底面左后前
    elements[:, :, :, 4] = base + stride_z         # 节点5: 顶面左下后
    elements[:, :, :, 5] = base + stride_x + stride_z   # 节点6: 顶面右下后
    elements[:, :, :, 6] = base + stride_x + stride_y + stride_z  # 节点7: 顶面右后后
    elements[:, :, :, 7] = base + stride_y + stride_z   # 节点8: 顶面左后后
    
    part = Part(nodes=nodes)

    node_per_elem = 8
    elem_type = f'C3D{node_per_elem}'

    if element_name is None:
        element_name = elem_type

    elems = initialize_element(element_type=elem_type, elems_index=torch.arange((nx-1)*(ny-1)*(nz-1)), elems=elements.reshape(-1, 8))
    
    part.add_element(elems, name=element_name)

    # surface_xmin: x=0 
    elem_indices_xmin = torch.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[0, :, :].reshape(-1)
    part.add_surface_set(
        name='surface_xmin', 
        elements=[(elem_indices_xmin, 5)]
    )

    # surface_xmax: x=nx-1 
    elem_indices_xmax = torch.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[-1, :, :].reshape(-1)
    part.add_surface_set(
        name='surface_xmax', 
        elements=[(elem_indices_xmax, 3)]
    )

    # surface_ymin: y=0 
    elem_indices_ymin = torch.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[:, 0, :].reshape(-1)
    part.add_surface_set(
        name='surface_ymin', 
        elements=[(elem_indices_ymin, 2)]
    )

    # surface_ymax: y=ny-1 
    elem_indices_ymax = torch.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[:, -1, :].reshape(-1)
    part.add_surface_set(
        name='surface_ymax', 
        elements=[(elem_indices_ymax, 4)]
    )

    # surface_zmin: z=0 
    elem_indices_zmin = torch.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[:, :, 0].reshape(-1)
    part.add_surface_set(
        name='surface_zmin', 
        elements=[(elem_indices_zmin, 0)]
    )

    # surface_zmax: z=nz-1 
    elem_indices_zmax = torch.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[:, :, -1].reshape(-1)
    part.add_surface_set(
        name='surface_zmax', 
        elements=[(elem_indices_zmax, 1)]
    )

    # define node sets for boundary conditions
    total_nodes = nx * ny * nz

    # 方法1：reshape 成网格形状再取边界
    nodes_grid = torch.arange(total_nodes).reshape(nx, ny, nz)

    # xmin 和 xmax: 固定 x 索引
    nodes_xmin = nodes_grid[0, :, :].reshape(-1)      # x=0 的所有节点
    nodes_xmax = nodes_grid[-1, :, :].reshape(-1)     # x=nx-1 的所有节点

    # ymin 和 ymax: 固定 y 索引
    nodes_ymin = nodes_grid[:, 0, :].reshape(-1)      # y=0 的所有节点
    nodes_ymax = nodes_grid[:, -1, :].reshape(-1)     # y=ny-1 的所有节点

    # zmin 和 zmax: 固定 z 索引
    nodes_zmin = nodes_grid[:, :, 0].reshape(-1)      # z=0 的所有节点
    nodes_zmax = nodes_grid[:, :, -1].reshape(-1)     # z=nz-1 的所有节点

    part.add_node_set(name='nodes_xmin', node_indices=nodes_xmin)
    part.add_node_set(name='nodes_xmax', node_indices=nodes_xmax)
    part.add_node_set(name='nodes_ymin', node_indices=nodes_ymin)
    part.add_node_set(name='nodes_ymax', node_indices=nodes_ymax)
    part.add_node_set(name='nodes_zmin', node_indices=nodes_zmin)
    part.add_node_set(name='nodes_zmax', node_indices=nodes_zmax)

    return part