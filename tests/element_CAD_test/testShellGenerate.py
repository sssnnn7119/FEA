import torch
import os
import numpy as np
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FEA
from FEA.elements import materials
from FEA.generate_shell import generate_shell_from_surface, add_shell_elements_to_model
from FEA.convert_elements import convert_elements_batch, convert_to_second_order
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

fem = FEA.FEA_INP()

def init_FEA(inp: FEA.FEA_INP,
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

   
    return fe


def generate_shell_model(fe: FEA.Main.FEA_Main, surface_names, shell_thickness: float) -> FEA.Main.FEA_Main:
    """
    Generate a new FEA model with C3D6 shell elements extruded from surfaces.
    
    Args:
        fe (FEA.Main.FEA_Main): The original FEA model
        surface_names (str or list): Name(s) of the surface(s) to extrude
        shell_thickness (float): Thickness of the extruded shell
        
    Returns:
        FEA.Main.FEA_Main: New FEA model with the original elements and the new shell elements
    """    # Generate shell elements from the surfaces
    nodes_new, c3d6_elements, c3d6_indices, offset_surface_sets = generate_shell_from_surface(
        fe, surface_names, shell_thickness)
    
    # Add the shell elements to create a new model
    new_fe = add_shell_elements_to_model(fe, nodes_new, c3d6_elements, c3d6_indices, offset_surface_sets)
    
    return new_fe


def test_convert_to_second_order(fe, element_names_to_convert=None):
    """
    Test the conversion of first-order elements to second-order elements.
    
    Args:
        fe (FEA.Main.FEA_Main): The FEA model to convert
        element_names_to_convert (list, optional): List of element set names to convert
        
    Returns:
        FEA.Main.FEA_Main: New FEA model with second-order elements
    """
    print("\nTesting conversion to second-order elements...")
    print(f"Original model nodes: {fe.nodes.shape[0]}")
    
    for elem_name, elem_obj in fe.elems.items():
        elem_type = type(elem_obj).__name__
        node_count = elem_obj._elems.shape[1] if hasattr(elem_obj, '_elems') else 0
        print(f"  {elem_name}: {elem_type}, {node_count} nodes per element")
    
    # Convert elements to second-order
    second_order_fe = convert_elements_batch(fe, element_names_to_convert)
    
    print("\nAfter conversion:")
    print(f"Node count: {second_order_fe.nodes.shape[0]}")
    
    for elem_name, elem_obj in second_order_fe.elems.items():
        elem_type = type(elem_obj).__name__
        node_count = elem_obj._elems.shape[1] if hasattr(elem_obj, '_elems') else 0
        print(f"  {elem_name}: {elem_type}, {node_count} nodes per element")
    
    return second_order_fe


# Main execution
if __name__ == "__main__":
    # Read the input file
    inp_file = os.path.join(current_path, '_forshellgenerate.inp')
    fem.Read_INP(inp_file)
    
    # Initialize the FEA model
    fe = init_FEA(fem)
    
    print("Original model stats:")
    print(f"Nodes: {fe.nodes.shape[0]}")
    print(f"Element groups: {len(fe.elems)}")
    for elem_name, elem_obj in fe.elems.items():
        print(f"  {elem_name}: {elem_obj._elems.shape[0]} elements")
      # Set shell thickness (can be adjusted as needed)
    shell_thickness = 0.5
    
    # Get all available surface names
    available_surfaces = [name for name in fe.surface_sets.keys() 
                         if name.startswith('surface_') and name.endswith('_All')]  # Exclude exterior surface
    
    print(f"Available surfaces for shell generation: {available_surfaces}")
    
    # Generate new model with shell elements from multiple surfaces
    # You can specify a single surface: 'surface_1_All'
    # Or multiple surfaces: ['surface_1_All', 'surface_2_All', ...]
    new_fe = generate_shell_model(fe, available_surfaces, shell_thickness)
    
    print("\nNew model with shell elements:")
    print(f"Nodes: {new_fe.nodes.shape[0]}")
    print(f"Element groups: {len(new_fe.elems)}")
    for elem_name, elem_obj in new_fe.elems.items():
        print(f"  {elem_name}: {elem_obj._elems.shape[0] if hasattr(elem_obj, '_elems') else 0} elements")
      # Convert both original and shell elements to second-order elements
    print("\nConverting elements to second-order...")
    # Convert all element sets to second-order (C3D4->C3D10, C3D6->C3D15)
    element_names = list(new_fe.elems.keys())
    print(f"Element sets to convert: {element_names}")
    
    # Use the batch conversion function to convert all elements
    second_order_fe = convert_elements_batch(new_fe, element_names)
    
    print("\nAfter conversion to second-order elements:")
    print(f"Nodes: {second_order_fe.nodes.shape[0]}")
    print(f"Element groups: {len(second_order_fe.elems)}")
    for elem_name, elem_obj in second_order_fe.elems.items():
        print(f"  {elem_name}: {elem_obj._elems.shape[0] if hasattr(elem_obj, '_elems') else 0} elements, node count per element: {elem_obj._elems.shape[1] if hasattr(elem_obj, '_elems') else 0}")
    
    # Optional: Visualize the model with mayavi
    try:
        from mayavi import mlab
        # Get surface elements for visualization
        # Combine all surfaces that were used for shell generation
        c3d4_surfaces = []
        for surface_name in available_surfaces:
            surface_elems = fe.get_surface_triangles(surface_name)
            if isinstance(surface_elems, list):
                c3d4_surfaces.extend(surface_elems)
            else:
                c3d4_surfaces.append(surface_elems)
        
        c3d4_surface = torch.cat(c3d4_surfaces, dim=0) if c3d4_surfaces else None
        
        # Get both top and bottom surfaces of the shell elements
        c3d6_surface_bottom = new_fe.elems['shell_elements'].find_surface(0, new_fe.elems['shell_elements']._elems_index)
        c3d6_surface_top = new_fe.elems['shell_elements'].find_surface(1, new_fe.elems['shell_elements']._elems_index)
        
        # Original surface and shell surfaces
        mlab.figure(1, bgcolor=(1, 1, 1))
        
        # Original surfaces
        # if c3d4_surface is not None:
        #     mlab.triangular_mesh(
        #         fe.nodes[:, 0].cpu().numpy(),
        #         fe.nodes[:, 1].cpu().numpy(),
        #         fe.nodes[:, 2].cpu().numpy(),
        #         c3d4_surface.cpu().numpy(),
        #         color=(0.7, 0.7, 1.0),
        #         opacity=0.3
        #     )

        # Shell bottom surface (matches original surface)
        mlab.triangular_mesh(
            new_fe.nodes[:, 0].cpu().numpy(),
            new_fe.nodes[:, 1].cpu().numpy(),
            new_fe.nodes[:, 2].cpu().numpy(),
            c3d6_surface_bottom.cpu().numpy(),
            color=(0.7, 1.0, 0.7),
            opacity=0.5
        )
        
        # Shell top surface (offset)
        mlab.triangular_mesh(
            new_fe.nodes[:, 0].cpu().numpy(),
            new_fe.nodes[:, 1].cpu().numpy(),
            new_fe.nodes[:, 2].cpu().numpy(),
            c3d6_surface_top.cpu().numpy(),
            color=(1.0, 0.7, 0.7),
            opacity=0.5
        )
        
        mlab.show()
    except ImportError:
        print("Mayavi not available for visualization")
    
    # Test the dedicated element conversion function
    print("\n==== Testing Element Conversion ====")
    print("1. Converting original C3D4 elements to C3D10...")
    original_to_second = test_convert_to_second_order(fe, list(fe.elems.keys()))
    
    print("\n2. First generating shell, then converting all elements...")
    shell_model = generate_shell_model(fe, available_surfaces, shell_thickness)
    print("\nGenerated shell model:")
    for elem_name, elem_obj in shell_model.elems.items():
        elem_type = type(elem_obj).__name__
        node_count = elem_obj._elems.shape[1] if hasattr(elem_obj, '_elems') else 0
        print(f"  {elem_name}: {elem_type}, {node_count} nodes per element")
    
    print("\n3. Converting the shell model entirely to second-order elements...")
    shell_to_second = test_convert_to_second_order(shell_model, list(shell_model.elems.keys()))
    
    c3d6_surface_bottom = shell_to_second.elems['shell_elements'].find_surface(0, shell_to_second.elems['shell_elements']._elems_index)
    c3d6_surface_top = shell_to_second.elems['shell_elements'].find_surface(1, shell_to_second.elems['shell_elements']._elems_index)
    
    shell_to_second.show_surface(name=['surface_1_All', 'surface_1_All_offset'])
    
    mlab.triangular_mesh(
        shell_to_second.nodes[:, 0].cpu().numpy(),
        shell_to_second.nodes[:, 1].cpu().numpy(),
        shell_to_second.nodes[:, 2].cpu().numpy(),
        c3d6_surface_bottom.cpu().numpy(),
        color=(0.7, 1.0, 0.7),
        opacity=0.5
    )
    
    # Shell top surface (offset)
    mlab.triangular_mesh(
        shell_to_second.nodes[:, 0].cpu().numpy(),
        shell_to_second.nodes[:, 1].cpu().numpy(),
        shell_to_second.nodes[:, 2].cpu().numpy(),
        c3d6_surface_top.cpu().numpy(),
        color=(1.0, 0.7, 0.7),
        opacity=0.5
    )
    
    mlab.show()

