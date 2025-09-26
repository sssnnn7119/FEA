import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')

import FEA
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)

fem = FEA.FEA_INP()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )
 
fem.read_inp(current_path + '/C3D4Less.inp')

fe = FEA.from_inp(fem)
fe.solver = FEA.solver.DynamicImplicitSolver(deltaT=1e-2, time_end=0.5)

fe.assembly.add_load(FEA.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06))

bc_dof = np.array(
    list(fem.part['final_model'].sets_nodes['surface_0_Bottom']))
bc_name = fe.assembly.add_constraint(
    FEA.constraints.Boundary_Condition(instance_name='final_model', index_nodes=bc_dof))

# rp = fe.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 80]))

# indexNodes = fem.part['final_model'].sets_nodes['surface_0_Head']

# fe.assembly.add_constraint(FEA.constraints.Couple(instance_name='final_model', indexNodes=indexNodes, rp_name=rp))

fe.initialize()

t1 = time.time()


benchmark_data = np.load('Z:/temp/benchmark_result.npy').astype(np.float64)
v_list = np.load('Z:/temp/velocity.npy').astype(np.float64)

# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
extern_surf = fe.assembly.get_instance('final_model').surfaces.get_elements('surface_0_All')[0]._elems.cpu().numpy()

# Ensure extern_surf is a triangle index array (n, 3)
if extern_surf.shape[1] > 3:
    extern_surf = extern_surf[:, :3]
elif extern_surf.shape[1] < 3:
    raise ValueError("Surface element array does not have enough vertices per face for triangles.")

from mayavi import mlab
import vtk
from mayavi import mlab
from time import sleep
import threading
coo=extern_surf

# Get the deformed surface coordinates

# Prepare undeformed surface
undeformed_surface = (fem.part['final_model'].nodes[:,1:]).cpu().numpy()
Unorm_list = []
deformed_list = []

for i in range(len(benchmark_data)):
    U = fe.assembly._GC2RGC(torch.from_numpy(benchmark_data[i]))[0].cpu().numpy()
    deformed_surface = undeformed_surface + U
    Unorm = (U**2).sum(axis=1)**0.5
    deformed_list.append(deformed_surface.copy())
    Unorm_list.append(Unorm.copy())

mlab.figure(bgcolor=(1,1,1), size=(800, 1000))

fig = mlab.gcf()
scene = fig.scene
scene.camera.parallel_projection = True

mesh = mlab.triangular_mesh(
    deformed_list[0][:, 0], deformed_list[0][:, 1], deformed_list[0][:, 2],
    extern_surf, scalars=Unorm_list[0]
)
mlab.view(azimuth=90, elevation=90)
scene.camera.parallel_projection = True
scene.camera.parallel_scale *= 2.

lut_mgr = mesh.module_manager.scalar_lut_manager

# 添加帧编号文本，设置为黑色
frame_text = mlab.text(0.01, 0.95, 'Frame: 0', width=0.2, color=(0,0,0))
for idx in range(len(benchmark_data)):
    deformed_surface = deformed_list[idx]
    Unorm = Unorm_list[idx]
    mesh.mlab_source.set(
        x=deformed_surface[:, 0],
        y=deformed_surface[:, 1],
        z=deformed_surface[:, 2],
        scalars=Unorm
    )
    lut_mgr.data_range = (0, 134)
    filename = f'Z:/temp/frames/frame_{idx:04d}.png'
    mlab.savefig(filename)
    frame_text.text = f'Frame: {idx}'

import imageio
import glob

def create_mp4_from_frames(frames_folder, output_path, fps=10):
    """
    从图像帧序列创建MP4视频。

    Args:
        frames_folder (str): 包含图像帧的目录路径。
        output_path (str): 输出MP4文件的保存路径。
        fps (int): 输出视频的每秒帧数。
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # 创建一个写入器对象
    writer = imageio.get_writer(output_path, fps=fps)


    # 将每一帧添加到视频中
    for idx in range(len(benchmark_data)):
        filename = f'Z:/temp/frames/frame_{idx:04d}.png'
        image = imageio.imread(filename)
        writer.append_data(image)

    # 关闭写入器
    writer.close()
    print(f"视频已保存至 {output_path}")

# --- 调用函数创建视频 ---
frames_directory = 'Z:/temp/frames'
output_video_file = 'Z:/temp/animation.mp4'
frame_rate = 50  # 在这里调节帧率

# 确保在创建视频前所有Mayavi窗口都已关闭，以完成图像保存
mlab.close(all=True)

create_mp4_from_frames(frames_directory, output_video_file, fps=frame_rate)