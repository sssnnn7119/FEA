import numpy as np
import torch
from . import constraints, elements, loads
from .elements import materials
from .elements.C3 import surfaces
from .FEA_INP import FEA_INP
from .Main import FEA_Main
from .reference_points import ReferencePoint

def from_inp(inp: FEA_INP) -> FEA_Main:
    """
    Load a FEA model from an INP file.

    Args:
        inp (FEA_INP): An instance of the FEA_INP class.

    Returns:
        FEA_Main: An instance of the FEA_Main class with imported elements and sets.
    """

    part_name = list(inp.part.keys())[0]

    num_nodes = [0]
    nodes = []
    for part_name, part in inp.part.items():
        num_nodes.append(num_nodes[-1] + part.nodes.shape[0])
        nodes.append(part.nodes[:, 1:])
    nodes = torch.cat(nodes, dim=0)

    fe = FEA_Main(nodes)
    
    elems_num = [0]
    for i in range(len(inp.part)):
        part_name = list(inp.part.keys())[i]
        part = inp.part[part_name]
        elems = inp.part[part_name].elems
        elems_num_now = 0
        
        
        for key in list(elems.keys()):

            materials_type = inp.part[part_name].elems_material[elems[key][:, 0], 2].type(torch.int).unique()

            elems_num_now += elems[key].shape[0]
            for mat_type in materials_type:
                index_now = torch.where(inp.part[part_name].elems_material[elems[key][:, 0], 2].type(torch.int) == mat_type)

                materials_now = materials.initialize_materials(
                    materials_type=mat_type.item(),
                    materials_params=inp.part[part_name].elems_material[elems[key][:, 0]][index_now][:, 3:]
                )

                element_name = key

                elems_now = elements.initialize_element(
                            element_type=element_name,
                            elems_index=torch.from_numpy(elems[key][:, 0] + elems_num[i]),
                            elems=torch.from_numpy(elems[key][:, 1:] + num_nodes[i]),
                            )
                
                elems_now.set_materials(materials_now)
                elems_now.set_density(inp.part[part_name].elems_material[elems[key][:, 0]][index_now][:, 1])
                
                fe.add_element(elems_now)

        elems_num.append(elems_num[-1] + elems_num_now)
        
        # Import all sets (node sets, element sets, and surface sets) from the INP file
        # Import node sets from each part

        for set_name, nodes in part.sets_nodes.items():
            full_name = f"{set_name}"
            fe.add_node_set(full_name, np.array(list(nodes))+num_nodes[i])
                
        # Import element sets from each part

        for set_name, elems in part.sets_elems.items():
            full_name = f"{set_name}"
            fe.add_element_set(full_name, np.array(list(elems))+elems_num[i])
                
        # Import surface sets from each part

        for surface_name, surface in part.surfaces.items():
            full_name = f"{surface_name}"
            sf_now = []
            for sf in surface:
                sf_now.append((sf[0]+elems_num[i], sf[1]))
            fe.add_surface_set(full_name, sf_now)

    return fe