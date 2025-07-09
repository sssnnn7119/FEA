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

    fe = FEA_Main(inp.part[part_name].nodes[:, 1:])
    elems = inp.part[part_name].elems
    
    for key in list(elems.keys()):

        materials_type = inp.part[part_name].elems_material[elems[key][:, 0], 2].type(torch.int).unique()
        for mat_type in materials_type:
            index_now = torch.where(inp.part[part_name].elems_material[elems[key][:, 0], 2].type(torch.int) == mat_type)

            materials_now = materials.initialize_materials(
                materials_type=mat_type.item(),
                materials_params=inp.part[part_name].elems_material[elems[key][:, 0]][index_now][:, 3:]
            )

            element_name = key

            elems_now = elements.initialize_element(
                        element_type=element_name,
                        elems_index=torch.from_numpy(elems[key][:, 0]),
                        elems=torch.from_numpy(elems[key][:, 1:]),
                        nodes=inp.part[part_name].nodes[:, 1:],
                        )
            
            elems_now.set_materials(materials_now)
            elems_now.set_density(inp.part[part_name].elems_material[elems[key][:, 0]][index_now][:, 1])
            
            fe.add_element(elems_now)
    
    # Import all sets (node sets, element sets, and surface sets) from the INP file
    # Import node sets from each part
    for part_name, part in inp.part.items():
        for set_name, nodes in part.sets_nodes.items():
            full_name = f"{set_name}"
            fe.add_node_set(full_name, list(nodes))
            
    # Import element sets from each part
    for part_name, part in inp.part.items():
        for set_name, elems in part.sets_elems.items():
            full_name = f"{set_name}"
            fe.add_element_set(full_name, list(elems))
            
    # Import surface sets from each part
    for part_name, part in inp.part.items():
        for surface_name, surface in part.surfaces.items():
            full_name = f"{surface_name}"
            fe.add_surface_set(full_name, surface)
    
    return fe