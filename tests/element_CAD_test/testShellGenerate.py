import torch
import os
import numpy as np
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FEA
from FEA.elements import materials
from FEA.elements.generate_shell import generate_shell_from_surface, add_shell_elements_to_model
from FEA.elements.convert_elements import convert_to_second_order
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)


def init_FEA(inp: FEA.FEA_INP,
                shell_thickness: float,
                shell_mu: float,
                shell_kappa: float,
                shell_density: float,
                surface_names: list[str],
                mu: dict[str, np.ndarray] = None,
                kappa: dict[str, np.ndarray] = None,
                density: dict[str, np.ndarray] = None) -> FEA.Main.FEA_Main:
    """
    Initialize the FEA class with the given input parameters.

    Parameters:
        inp (FEA.FEA_INP): The input parameters for the FEA class.

    Returns:
        FEA.Main.FEA_Main: An instance of the FEA_Main class with the given input parameters.
        
    """
    fe = FEA.from_inp(inp)

    fe.maximum_iteration = 100

    num_thick_element = 4

    surface_names_list = [surface_names]
    for i in range(num_thick_element):
        
        if i < num_thick_element - 1:
            surface_new_name = [surface_names[k] + ('_offset%d'%i) for k in range(len(surface_names))]
        else:
            surface_new_name = [surface_names[k] + '_offset' for k in range(len(surface_names))]
        surface_names_list.append(surface_new_name)

        nodes_new, c3d6_elements, c3d6_indices, offset_surface_sets = FEA.elements.generate_shell_from_surface(
            fe=fe, surface_names=surface_names_list[i], shell_thickness=shell_thickness/num_thick_element, surface_new_name=surface_names_list[i+1])

        # Add the shell elements to create a new model
        fe = FEA.elements.add_shell_elements_to_model(
            fe=fe, nodes_new=nodes_new, c3d6_elements=c3d6_elements,c3d6_indices= c3d6_indices, name_new_elements='shell_elements%d'%i, offset_surface_sets=offset_surface_sets)
    
    if num_thick_element > 2:
        fe.merge_elements(element_name_list=['shell_elements%d'%i for i in range(num_thick_element-1)], element_name_new='shell_elements')
    else:
        fe.elems['shell_elements'] = fe.elems['shell_elements0']
        del fe.elems['shell_elements0']


    
    # fe.elems['shell_elements'] = fe.elems['shell_elements0']
    # del fe.elems['shell_elements0']
    fe.elems['pressure_elements'] = fe.elems['shell_elements%d' % (num_thick_element - 1)]
    del fe.elems['shell_elements%d' % (num_thick_element - 1)]


    # convert the elements to C3D10 and C3D15
    # element_names_to_convert = list(fe.elems.keys())
    # fe = FEA.elements.convert_to_second_order(fe,
    #                                  element_names_to_convert)

    
    fe = FEA.elements.convert_to_second_order(
        fe, element_names=['pressure_elements'])
    
    new_elems = FEA.elements.C3.C3D15Transition12(elems=fe.elems['pressure_elements']._elems,
                                                    elems_index=fe.elems['pressure_elements']._elems_index,)
    fe.elems['pressure_elements'] = new_elems

    # assign the materials to the elements
    if mu is not None and kappa is not None and density is not None:
        for str_now in mu.keys():
            materials_now = materials.NeoHookean(
                mu=torch.from_numpy(mu[str_now]).to(fe.nodes.device).to(
                    fe.nodes.dtype),
                kappa=torch.from_numpy(kappa[str_now]).to(
                    fe.nodes.device).to(fe.nodes.dtype),
            )

            fe.elems[str_now].set_density(
                torch.from_numpy(density[str_now]).to(fe.nodes.device).to(
                    fe.nodes.dtype))
            fe.elems[str_now].set_materials(materials_now)

    # assign the shell material to the elements
    materials_shell = materials.NeoHookean(
        mu=torch.tensor(shell_mu,
                        dtype=torch.float64,
                        device=fe.nodes.device),
        kappa=torch.tensor(shell_kappa,
                            dtype=torch.float64,
                            device=fe.nodes.device),
    )
    fe.elems['shell_elements'].set_density(
        torch.tensor(shell_density,
                        dtype=torch.float64,
                        device=fe.nodes.device))
    fe.elems['shell_elements'].set_materials(materials_shell)

    fe.elems['pressure_elements'].set_density(
        torch.tensor(shell_density,
                        dtype=torch.float64,
                        device=fe.nodes.device))
    fe.elems['pressure_elements'].set_materials(materials_shell)
    
    # add loads
    i = 0
    while True:
        if 'surface_%d_All_offset' % (i + 1) not in fe.surface_sets.keys():
            break
        fe.add_load(FEA.loads.Pressure(
            surface_set='surface_%d_All_offset' % (i + 1), pressure=0.),
                    name='Pressure_%d' % i)
        i += 1

    # add boundary condition
    bc_dof = np.where((abs(fe.nodes[:, 2] - 0)
                            < 0.1).cpu().numpy())[0] * 3
    bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
    fe.add_constraint(FEA.constraints.Boundary_Condition(indexDOF=bc_dof,
                                                            dispValue=0.),
                        name='BC')

    # add reference point and constraints
    rp = FEA.ReferencePoint([0., 0., fe.nodes[:, 2].max()], )
    rp_name = fe.add_reference_point(rp=rp)
    indexNodes = np.where((abs(fe.nodes[:, 2] - rp.node[2])
                            < 0.1).cpu().numpy())[0]
    fe.add_constraint(
        FEA.constraints.Couple(indexNodes=indexNodes, rp_name=rp_name))

    mid_nodes_index = fe.elems['element-0'].get_2nd_order_point_index()
    fe.nodes[mid_nodes_index[:,
                                0]] = (fe.nodes[mid_nodes_index[:, 1]] +
                                    fe.nodes[mid_nodes_index[:, 2]]) / 2.0

    return fe

if __name__ == '__main__':
    fem = FEA.FEA_INP()
