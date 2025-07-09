import torch
import os
import numpy as np
import time
import sys
sys.path.append(os.getcwd())

import FEA

current_path = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)

def fea_inp(inp_name: str):
    fem = FEA.FEA_INP()
    # fem.Read_INP(
    #     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
    # )

    # fem.Read_INP(
    #     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
    # )

    fem.Read_INP(current_path + '/'+inp_name+'.inp')

    fe = FEA.from_inp(fem)

    fe.elems['element-0'].surf_order[:] = 1.

    bc_dof = np.where((abs(fe.nodes[:, 2] - 0)
                            < 0.1).cpu().numpy())[0] * 3
    bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
    fe.add_constraint(
        FEA.constraints.Boundary_Condition(indexDOF=bc_dof,
                                        dispValue=torch.zeros(bc_dof.size)))

    rp = fe.add_reference_point(FEA.ReferencePoint([0, 0, 100]))
    indexNodes = np.where((abs(fe.nodes[:, 2] - 100)
                            < 0.1).cpu().numpy())[0]


    fe.add_constraint(FEA.constraints.Couple(indexNodes=indexNodes, rp_name=rp))
    fe.add_load(
        FEA.loads.Concentrate_Force(rp_name=rp, force=[50., 0., 0.]))
    t1 = time.time()
    fe.initialize()

    fe.solve(tol_error=0.000001)

    # mid_nodes_index = fe.elems['element-0'].get_2nd_order_point_index()
    # elems = fe.elems['element-0']

    # fe.nodes[mid_nodes_index[:, 0]] = (fe.nodes[mid_nodes_index[:, 1]] + fe.nodes[mid_nodes_index[:, 2]]) / 2.0
    # fe.solve(RGC0=fe.RGC, tol_error=0.01)
    # if fe.elems['element-0'].order == 1:
    #     fe.solve(tol_error=0.01)
    # else:
    #     fe.elems['element-0'].set_order(1)
    #     fe.solve(tol_error=0.01)

    #     fe.elems['element-0'].set_order(2)
    #     fe.refine_RGC()
    #     fe.solve(RGC0=fe.RGC, tol_error=0.01)

    print(fe.GC[-6:])
    print('ok')
    return fe.GC[-6:].tolist()


element_list = ['C3D10', 'C3D15', 'C3D20', 'C3D4', 'C3D6', 'C3D8', ]

results = {}
for element in element_list:
    print(f'Running test for element: {element}')
    results[element] = fea_inp(element)
    
print("All tests completed.")
print("Results:")
for element, result in results.items():
    print(f"{element}\t: {result}")

# from mayavi import mlab
# import vtk
# from mayavi import mlab

# # # Get the deformed surface coordinates
# U = fe.RGC[0].cpu().numpy()
# undeformed_surface = fe.nodes.cpu().numpy()
# deformed_surface = undeformed_surface + U

# r=deformed_surface.transpose()


# Unorm = (U**2).sum(axis=1)**0.5

# # surface = mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo)
# # surface_vtk = surface.outputs[0]._vtk_obj
# # stlWriter = vtk.vtkSTLWriter()
# # stlWriter.SetFileName('test.stl')
# # stlWriter.SetInputConnection(surface_vtk.GetOutputPort())
# # stlWriter.Write()
# # mlab.close()

# # Plot the deformed surface
# mlab.points3d(deformed_surface[:, 0], deformed_surface[:, 1], deformed_surface[:, 2], scale_factor=0.4)


# node_i = deformed_surface[fe.elems['element-0']._elems[500].cpu().numpy()]
# mlab.points3d(node_i[:, 0], node_i[:, 1], node_i[:, 2], scale_factor=0.5, color=(1, 0, 0))

# mlab.show()