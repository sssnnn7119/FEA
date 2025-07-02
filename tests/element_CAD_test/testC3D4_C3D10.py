import torch
import os
import numpy as np
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FEA
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

fem = FEA.FEA_INP()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )

fem.Read_INP(current_path + '/C3D4.inp')

fe = FEA.from_inp(fem)

nodes_new, element_new = fe.elems['element-0'].to_C3D10(nodes_now=fe.nodes.clone())

fe.nodes = nodes_new
element_delete = fe.elems['element-0']
fe.elems['element-0'] = element_new

mid_nodes_index = fe.elems['element-0'].get_2nd_order_point_index()



fe.add_load(FEA.loads.Pressure(surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')

bc_dof = np.where((abs(fe.nodes[:, 2] - 0)
                        < 0.1).cpu().numpy())[0] * 3
# bc_dof = np.array(
#     list(fem.part['final_model'].sets_nodes['surface_0_Bottom'])) * 3
bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
bc_name = fe.add_constraint(
    FEA.constraints.Boundary_Condition(indexDOF=bc_dof,
                                    dispValue=torch.zeros(bc_dof.size)))

rp = fe.add_reference_point(FEA.ReferencePoint([0, 0, 80]))

indexNodes = np.where((abs(fe.nodes[:, 2] - 80)
                        < 0.1).cpu().numpy())[0]

fe.add_constraint(FEA.constraints.Couple(indexNodes=indexNodes, rp_name=rp))



t1 = time.time()


fe.elems['element-0'].set_order(1)
fe.solve(tol_error=0.01)
# if fe.elems['element-0'].order == 1:
#     fe.solve(tol_error=0.01)
# else:
#     fe.elems['element-0'].set_order(1)
#     fe.solve(tol_error=0.01)

#     fe.elems['element-0'].set_order(2)
#     fe.refine_RGC()
#     RGC1 = fe.RGC
#     fe.solve(RGC0=fe.RGC, tol_error=0.01)


print(fe.GC)
print('ok')


# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
extern_surf = fem.Find_Surface(['surface_0_All'])[1]
# extern_surf = fem.part['final_model'].surfaces['surface_1_All']

from mayavi import mlab
import vtk
from mayavi import mlab
coo=extern_surf

# Get the deformed surface coordinates
U = fe.RGC[0].cpu().numpy()
undeformed_surface = (fe.nodes).cpu().numpy()
deformed_surface = undeformed_surface + U

r=deformed_surface.transpose()


Unorm = (U**2).sum(axis=1)**0.5

# surface = mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo)
# surface_vtk = surface.outputs[0]._vtk_obj
# stlWriter = vtk.vtkSTLWriter()
# stlWriter.SetFileName('test.stl')
# stlWriter.SetInputConnection(surface_vtk.GetOutputPort())
# stlWriter.Write()
# mlab.close()

# Plot the deformed surface
mlab.triangular_mesh(deformed_surface[:, 0], deformed_surface[:, 1], deformed_surface[:, 2], extern_surf, scalars=Unorm)
mlab.show()