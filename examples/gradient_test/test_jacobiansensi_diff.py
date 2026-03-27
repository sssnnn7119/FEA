
import torch
import os
import time
import sys

sys.path.append('.')
import torchfea
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

from typing import Optional, TYPE_CHECKING, Callable

def get_jacobian_sensitivity(
    self: torchfea.solver.StaticImplicitSolver,
    fe_result: torchfea.solver.StaticResult,
    fe_result2: Optional[torchfea.solver.StaticResult],
    fe_resultdp: Optional[torchfea.solver.StaticResult],
    design_vars: torch.Tensor,
    load_names: Optional[list[str]],
    apply_func: Callable[[torchfea.Assembly, torch.Tensor], None] ,
    compute_objective_func: Callable[[torch.Tensor, torch.Tensor, torchfea.Assembly, dict[str, torch.Tensor]], torch.Tensor],
    other_args: Optional[dict[str, torch.Tensor]] = None,
    jacobian_dict: Optional[dict[str, torch.Tensor]] = None
) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Jacobian sensitivity (dR/dVars) for the static problem.

        This function computes the sensitivity of the residual forces with respect to design variables,
        which is essential for gradient-based optimization and design.

        Args:
            fe_result (StaticResult): The result object containing the current state and factorized stiffness matrix.
            design_vars (torch.Tensor): A tensor representing the design variables.
                It must be the source of gradients for `apply_func`.
            load_names (list[str], optional): The names of the loads for which to compute the Jacobian sensitivity. If None, all loads are considered.
            apply_func (Callable[[Assembly, torch.Tensor], None]): 
                A callback to apply design variables to the assembly.
                - Signature: `def apply_func(assembly: Assembly, design_vars: torch.Tensor) -> None`
                - Behavior: Modify `assembly` in-place using `design_vars`. Operations must be traceable
                by Autograd (e.g., `part.nodes = original_nodes + design_vars.reshape_as(part.nodes)`).
            compute_objective_func (Callable[[torch.Tensor, torch.Tensor, Assembly, dict[str, torch.Tensor]], torch.Tensor]): 
                A callback to compute the objective scalar.
                - Signature: `def compute_objective_func(GC: torch.Tensor, jacobian: torch.Tensor, assembly: Assembly, other_args: Optional[dict[str, torch.Tensor]]) -> torch.Tensor`
                - Args: `GC` is the displacement vector (detached from physics but tracking gradient).
                - Returns: A scalar tensor representing the objective value (e.g., compliance, stress).
            other_args (dict[str, torch.Tensor], optional): Additional arguments for the objective function.
            jacobian_dict (dict[str, torch.Tensor], optional): If provided, this dictionary will be calculated and returned as the Jacobian sensitivities (dR/dVars) for each load. If None, only the objective gradient will be computed.
        Returns:
            tuple[dict[str, torch.Tensor], torch.Tensor]:
                - jacobian_dict: The Jacobian matrix dGC/dVars, where GC is generalized coordinates and Vars are design variables.
                    jacobian_dict.values(): (num_dofs, num_load_params).
                - jacobian_sensitivity: Second-order sensitivity dL_j/dPara dVars for the objective function L_j with respect to load parameters Para and design variables.
                    Shape: design_vars.shape.
        """
    # try:
        # 0. Set load parameters and prepare jacobian
        self.assembly.set_load_parameters(fe_result.load_params)

        # 1. Factorize system if needed
        if fe_result.if_factorized is False:
            fe_result.factorize_stiffness_matrix(assembly=self.assembly)

        # 2. Prepare Jacobian
        if jacobian_dict is None:
            jacobian_dict = self.get_jacobian(fe_result, load_names=load_names)

        jacobian = torch.cat([jacobian_dict[load_name] for load_name in load_names], dim=1)

        # 3. Prepare Autograd graph
        design_vars_grad = design_vars.clone().detach().requires_grad_(True)
        GC_grad = fe_result.GC.clone().detach().requires_grad_(True)
        jacobian_grad = jacobian.clone().detach().requires_grad_(True)
        
        # 4. Apply Design Variables
        apply_func(self.assembly, design_vars_grad)
        self.assembly.initialize()
        R_grad = self.assembly.assemble_force(GC=fe_result.GC)

        # 5. Compute first adjoint vector W0
        if other_args is None:
            objective = compute_objective_func(GC_grad, jacobian_grad, self.assembly)
        else:
            objective = compute_objective_func(GC_grad, jacobian_grad, self.assembly, other_args)

        # 6. Get Ldx and Ldy for the adjoint solve
        objective.backward(retain_graph=True)
        if GC_grad.grad is None:
            Ldx = torch.zeros_like(GC_grad)
        else:
            Ldx = GC_grad.grad.clone().detach()
        if jacobian_grad.grad is None:
            Ldy = torch.zeros_like(jacobian_grad)
        else:
            Ldy = jacobian_grad.grad.clone().detach()

        # 7. sensitivity for GC
        W0 = -fe_result.K_solver.solve(fe_result.K_sp, Ldx.cpu().numpy())
        W0_tensor = torch.tensor(W0, dtype=GC_grad.dtype, device=GC_grad.device)
        obj_part_x = (W0_tensor * R_grad).sum()

        # 8. For each parameter, compute the Jacobian sensitivity using the chain rule:

        ## get the current load parameters as a single tensor
        total_load_params_list = []
        for load_name in load_names:
            load = self.assembly._loads[load_name]
            total_load_params_list.append(load._parameters.flatten())
        total_load_params = torch.cat(total_load_params_list, dim=0)
        num_load_params = total_load_params.numel()

        from torch.autograd.functional import jvp
        obj_part_y = torch.zeros(1, device=GC_grad.device, dtype=GC_grad.dtype)
        K_indices = self.assembly.assemble_Stiffness_Matrix(GC=fe_result.GC)[1]
        wKdp = torch.zeros_like(fe_result.GC)
        for para_idx in range(num_load_params):

            
            ### compute the Jacobian sensitivity using the chain rule:
            Ldy_now = Ldy[:, para_idx]
            W1 = fe_result.K_solver.solve(fe_result.K_sp, Ldy_now.cpu().numpy())
            W1_tensor = torch.tensor(W1, dtype=GC_grad.dtype, device=GC_grad.device)

            ### evaluate the stiffness matrix sensitivity dK/dPara using autograd
            def get_Kdp(load_now: torch.Tensor):
                GC_now = fe_result.GC + jacobian[:, para_idx] * (load_now - load_now.detach())
                index_now = 0
                for load_name in load_names:
                    load = self.assembly._loads[load_name]
                    param_len = load._parameters.numel()
                    idx_now = para_idx - index_now
                    if 0 <= idx_now < param_len:
                        load._parameters[idx_now] = load_now
                        break
                    index_now += param_len
                K_indices, K_values = self.assembly.assemble_Stiffness_Matrix(GC=GC_now)[1:]
                return K_values
            Kdp_values = torch.autograd.functional.jvp(
                get_Kdp, total_load_params[para_idx: para_idx+1], torch.ones([1]), create_graph=False)[1]
            Kdp = torch.sparse_coo_tensor(K_indices, Kdp_values, size=fe_result.K_sp.shape)

            K_values = self.assembly.assemble_Stiffness_Matrix(GC=fe_result.GC)[2]
            Kp_values = self.assembly.assemble_Stiffness_Matrix(GC=fe_resultdp.GC)[2]

            wKdp += Kdp@ W1_tensor

            ### evaluate the stiffness matrix sensitivity dR/dPara using autograd
            def get_Rdp(load_now: torch.Tensor, GC=fe_result.GC):
                GC_now = GC + jacobian[:, para_idx] * (load_now - load_now.detach())
                index_now = 0
                for load_name in load_names:
                    load = self.assembly._loads[load_name]
                    param_len = load._parameters.numel()
                    idx_now = para_idx - index_now
                    if 0 <= idx_now < param_len:
                        load._parameters[idx_now] = load_now
                        break
                    index_now += param_len
                R = self.assembly.assemble_force(GC=GC_now)
                return (R * W1_tensor).sum()
            
            R1 = get_Rdp(total_load_params[para_idx: para_idx+1])
            Rp = get_Rdp(total_load_params[para_idx: para_idx+1] + epsilon / 100, GC=fe_resultdp.GC)

            obj_part_y -= torch.autograd.functional.jacobian(get_Rdp, total_load_params[para_idx: para_idx+1], create_graph=True)

        wKdpKinv = fe_result.K_solver.solve(fe_result.K_sp, wKdp.cpu().numpy())
        wKdpKinv_tensor = torch.tensor(wKdpKinv, dtype=GC_grad.dtype, device=GC_grad.device)
        obj_part_y += (wKdpKinv_tensor * R_grad).sum()

        obj_total = obj_part_x + obj_part_y
        obj_total.backward()
        sensitivity = design_vars_grad.grad.clone().detach()

    # finally:            
        # Cleanup: Detach all tensors in assembly to prevent graph explosion in next run
        self._detach_recursive(self.assembly)

        return jacobian_dict, sensitivity


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
                
# fe.assembly.add_load(torch_fea.loads.ContactSelf(surface_name='surface_0_All', penalty_distance_g=10, penalty_threshold_h=5.5))
# fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_0_All'))
# fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_1_All'))
# fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_2_All'))
# fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_3_All'))

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
    part.nodes[0, 2] = design_vars

def compute_objective(GC: torch.Tensor,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    return GC[-2]
    
grad_sensi = solver.get_sensitivity(
    fe_result=feresult,
    design_vars=part.nodes[0, 2].reshape(-1),
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective,
    )

jacobian = solver.get_jacobian(feresult, load_names=['pressure-1'])

def compute_objective_jacobian(GC: torch.Tensor,
                               jacobian: torch.Tensor,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    return jacobian[-2, 0] + GC[-2]

jacobian, grad_sensi_jacobian = solver.get_jacobian_sensitivity(
    fe_result=feresult,
    design_vars=part.nodes[0, 2].reshape(-1),
    load_names=['pressure-1'],
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective_jacobian,
    )
grad_sensi_jacobian = grad_sensi_jacobian
jacobian0 = torch.cat([jacobian[load_name] for load_name in ['pressure-1']], dim=1)
nodes0 = part.nodes.clone().detach()
epsilon = 1e-3
test_pair = ((2, 1), (10, 0), (5, 1))


obj0 = compute_objective_jacobian(GC0, jacobian0, fe.assembly)

indtest1 = 0
indtest2 = 2
# if (nodes0[indtest1, 2] != 70):
#     continue
part.nodes = nodes0.detach().clone()
part.nodes[indtest1, indtest2] += epsilon
result1 = fe.solve(tol_error=1e-6, GC0=GC0)
GC1 = fe.assembly.GC.clone().detach()
jacobian1 = solver.get_jacobian(result1, load_names=['pressure-1'])
jacobian1 = torch.cat([jacobian1[load_name] for load_name in ['pressure-1']], dim=1)
obj1 = compute_objective_jacobian(GC1, jacobian1, fe.assembly)



part.nodes = nodes0.detach().clone()
fe.assembly.get_load('pressure-1').pressure += epsilon / 100
resultpressure = fe.solve(tol_error=1e-6, GC0=GC0)


get_jacobian_sensitivity(
    solver,
    fe_result=feresult,
    fe_result2=result1,
    fe_resultdp=resultpressure,
    design_vars=part.nodes[0, 2].reshape(-1),
    load_names=['pressure-1'],
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective_jacobian,
    jacobian_dict=jacobian,
)


diff = (obj1 - obj0) / epsilon
print('Testing node (%d, %d):' % (indtest1, indtest2))
print('Finite difference Jacobian sensitivity:', diff.item())
print('Autograd Jacobian sensitivity:', grad_sensi_jacobian.item())
print('Error:', abs(diff - grad_sensi_jacobian.item()) / abs(diff))
print('\n\n')

print('Gradient check for node position:')
print('Autograd gradient:')


assert False