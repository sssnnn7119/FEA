import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')

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
# fe._maximum_step_length = 0.3
# elems = FEA.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# FEA.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=FEA.elems['C3D4']._elems_index))

fe.add_load(FEA.loads.Pressure(surface_set='surface_1_All', pressure=0.08),
                name='pressure-1')
# fe.add_load(FEA.loads.ContactSelf(surface_name='surface_0_All', penalty_distance_g=10, penalty_threshold_h=5.5))
fe.add_load(FEA.loads.ContactSelf(surface_name='surface_0_All'))
fe.add_load(FEA.loads.ContactSelf(surface_name='surface_1_All'))
fe.add_load(FEA.loads.ContactSelf(surface_name='surface_2_All'))
fe.add_load(FEA.loads.ContactSelf(surface_name='surface_3_All'))

bc_dof = np.array(
    list(fem.part['final_model'].sets_nodes['surface_0_Bottom'])) * 3
bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
bc_name = fe.add_constraint(
    FEA.constraints.Boundary_Condition(indexDOF=bc_dof,
                                    dispValue=torch.zeros(bc_dof.size)))

rp = fe.add_reference_point(FEA.ReferencePoint([0, 0, 70]))

indexNodes = np.where((abs(fe.nodes[:, 2] - 70)
                        < 0.1).cpu().numpy())[0]
# FEA.add_constraint(
#     Constraints.Couple(
#         indexNodes=indexNodes,
#         rp_index=2))
fe.add_constraint(FEA.constraints.Couple(indexNodes=indexNodes, rp_name=rp))



t1 = time.time()

fe.solve(tol_error=0.01)


print(fe.GC[-6:])
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
undeformed_surface = (fem.part['final_model'].nodes[:,1:]).cpu().numpy()
deformed_surface = undeformed_surface + U

r=deformed_surface.transpose()


Unorm = (U**2).sum(axis=1)**0.5

# Plot the deformed surface
# Plot the deformed surface with triangular faces
mesh = mlab.triangular_mesh(deformed_surface[:, 0], deformed_surface[:, 1], deformed_surface[:, 2], 
                           extern_surf, scalars=Unorm)

# Add the edges of each triangle
mesh.actor.property.edge_visibility = True
mesh.actor.property.line_width = 1.0
mesh.actor.property.edge_color = (0, 0, 0)  # Black edges
mlab.show()