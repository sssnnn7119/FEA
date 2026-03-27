import pypardiso
import torch
import os
import numpy as np
import time
import sys

from torch.nn import grad
sys.path.append('.')
import scipy.sparse as sp
import torchfea
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)


fem = torchfea.FEA_INP()
name = 'C3D4'
fem.read_inp(current_path + '/C3D4.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver()
# fe._maximum_step_length = 0.3
# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')
fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_2_All', pressure=0.02),
                name='pressure-2')
                
# fe.assembly.add_load(torch_fea.loads.ContactSelf(surface_name='surface_0_All', penalty_distance_g=10, penalty_threshold_h=5.5))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_0_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_1_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_2_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_3_All'))

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 70]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))


t1 = time.time()
fe.initialize()
if not os.path.exists('Z:/temp/%s_results.npz' % name):
    feresult = fe.solve(tol_error=1e-6)
    feresult.save('Z:/temp/%s_results.npz' % name)
else:
    feresult = torchfea.solver.StaticResult.load('Z:/temp/%s_results.npz' % name)
GC0 = feresult.GC.clone().detach()
RGC0 = fe.assembly._GC2RGC(GC0)



fe.initialize()

part = fe.assembly.get_part('final_model')
solver: torchfea.solver.StaticImplicitSolver = fe.solver
def apply_design_vars(assembly: torchfea.Assembly,
                        design_vars: torch.Tensor,
                        ) -> None:
    part = assembly.get_part('final_model')
    part.nodes = design_vars.reshape(part.nodes.shape)

jacobian = solver.get_jacobian(feresult, load_names=['pressure-1', 'pressure-2'])

def compute_objective_jacobian(GC: torch.Tensor,
                               jacobian: torch.Tensor,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    return jacobian[-2, 0] * GC[-2] * jacobian[-2, 1]

jacobian, grad_sensi_jacobian = solver.get_jacobian_sensitivity(
    fe_result=feresult,
    design_vars=part.nodes.reshape(-1),
    load_names=['pressure-1', 'pressure-2'],
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective_jacobian,
    )
grad_sensi_jacobian = grad_sensi_jacobian.reshape(part.nodes.shape)
jacobian0 = torch.cat([jacobian[load_name] for load_name in ['pressure-1', 'pressure-2']], dim=1)
nodes0 = part.nodes.clone().detach()
epsilon = 1e-3
test_pair = ((2, 1), (10, 0), (5, 1))

index_test = torch.where(grad_sensi_jacobian.abs() > 0.000001)

obj0 = compute_objective_jacobian(GC0, jacobian0, fe.assembly)

for i in range(index_test[0].shape[0]):
    indtest1 = index_test[0][i].item()
    indtest2 = index_test[1][i].item()
    # if (nodes0[indtest1, 2] != 70):
    #     continue
    part.nodes = nodes0.detach().clone()
    part.nodes[indtest1, indtest2] += epsilon
    result1 = fe.solve(tol_error=1e-6, GC0=GC0)
    GC1 = fe.assembly.GC.clone().detach()
    jacobian1 = solver.get_jacobian(result1, load_names=['pressure-1', 'pressure-2'])
    jacobian1 = torch.cat([jacobian1[load_name] for load_name in ['pressure-1', 'pressure-2']], dim=1)
    obj1 = compute_objective_jacobian(GC1, jacobian1, fe.assembly)

    diff = (obj1 - obj0) / epsilon
    print('Testing node (%d, %d):' % (indtest1, indtest2))
    print('Finite difference Jacobian sensitivity:', diff.item())
    print('Autograd Jacobian sensitivity:', grad_sensi_jacobian[indtest1, indtest2].item())
    print('Error:', abs(diff - grad_sensi_jacobian[indtest1, indtest2].item()) / abs(diff))
    print('\n\n')

print('Gradient check for node position:')
print('Autograd gradient:')


assert False