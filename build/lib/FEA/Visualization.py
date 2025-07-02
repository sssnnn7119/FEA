import numpy as np
from FEA_INP import FEA_INP

fem = FEA_INP()
fem.Read_INP(
    'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/TRO20230207Morph/20230411Revise/Result2/Twist400/FEA/c3d10.inp'
)

# fem.Read_INP(
#     'Z:\RESULT\T20240106165728_\Cache/TopOptRun.inp'
# )

# fem.Read_INP(
#     'ToolBox/FEA/case/TopOptRun.inp'
# )

# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

extern_surf = fem.Find_Surface(['lateral1'])[1]
# extern_surf = fem.Find_Surface(['surface_1_All'])[1]
# extern_surf = fem.part['final_model'].surfaces['surface_1_All']

from mayavi import mlab
import vtk
from mayavi import mlab
coo=extern_surf

# Get the deformed surface coordinates
U = np.loadtxt('tensor.txt')
undeformed_surface = (fem.part['final_model'].nodes[:,1:]).cpu().numpy()
deformed_surface = undeformed_surface + U

r=deformed_surface.transpose()


Unorm = (U**2).sum(axis=1)**0.5

surface = mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo)
surface_vtk = surface.outputs[0]._vtk_obj
stlWriter = vtk.vtkSTLWriter()
stlWriter.SetFileName('test.stl')
stlWriter.SetInputConnection(surface_vtk.GetOutputPort())
stlWriter.Write()
mlab.close()

# Plot the deformed surface
mlab.triangular_mesh(deformed_surface[:, 0], deformed_surface[:, 1], deformed_surface[:, 2], extern_surf, scalars=Unorm)

mlab.show()

