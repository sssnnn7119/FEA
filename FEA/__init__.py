import numpy as np
import torch
from .inp import FEA_INP
from .controller import FEAController
from .assemble import Part, Instance, ReferencePoint, Assembly
from .assemble import materials, elements, loads, constraints, surfaces
from . import solver


def from_inp(inp: FEA_INP, create_instance=True) -> FEAController:
    """
    Load a FEA model from an INP file.

    Args:
        inp (FEA_INP): An instance of the FEA_INP class.

    Returns:
        FEA_Main: An instance of the FEA_Main class with imported elements and sets.
    """

    assembly_now = Assembly()

    part_name = list(inp.part.keys())[0]

    for i in range(len(inp.part)):
        part_name = list(inp.part.keys())[i]
        part_nodes = inp.part[part_name]

        part_now = Part(part_nodes.nodes[:, 1:])

        assembly_now.add_part(part=part_now, name=part_name)
        if create_instance:
            assembly_now.add_instance(instance=Instance(part_now), name=part_name)

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
                            elems_index=torch.from_numpy(elems[key][:, 0]),
                            elems=torch.from_numpy(elems[key][:, 1:]),
                            part=part_now
                            )
                
                elems_now.set_materials(materials_now)
                elems_now.set_density(inp.part[part_name].elems_material[elems[key][:, 0]][index_now][:, 1])
                
                part_now.add_element(elems_now)
 
        # Import surface sets from each part
        for surface_name, surface in part_nodes.surfaces.items():
            full_name = f"{surface_name}"
            sf_now = []
            for sf in surface:
                sf_now.append((sf[0], sf[1]))
            part_now.add_surface_set(full_name, sf_now)


    fe = FEAController()
    fe.assembly = assembly_now
    return fe