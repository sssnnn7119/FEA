import os
import numpy as np
import torch
import FEA

# Configure dtype/device
torch.set_default_dtype(torch.float64)
# Uncomment to use GPU if available
# torch.set_default_device(torch.device('cuda'))


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    inp_path = os.path.join(repo_root, 'tests', 'pressure_test', 'C3D4Less.inp')

    # 1) Read INP
    fem = FEA.FEA_INP()
    fem.read_inp(inp_path)

    # 2) Build controller from INP
    controller = FEA.from_inp(fem)

    # 3) Add load: surface pressure on surface_1_All
    controller.assembly.add_load(
        FEA.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
        name='pressure-1'
    )

    # 4) Add boundary: fix bottom nodes (Dirichlet)
    bc_nodes = np.array(list(fem.part['final_model'].sets_nodes['surface_0_Bottom']))
    controller.assembly.add_boundary(
        FEA.boundarys.Boundary_Condition(instance_name='final_model', index_nodes=bc_nodes)
    )

    # 5) Solve with static implicit
    controller.solver = FEA.solver.StaticImplicitSolver()
    controller.solve(tol_error=1e-2)

    # 6) Save displacement vector
    out_dir = os.path.join(repo_root, 'out')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'static_pressure_GC.npy'), controller.solver.GC.cpu().numpy())

    print('Done. Saved GC to', os.path.join(out_dir, 'static_pressure_GC.npy'))


if __name__ == '__main__':
    main()
