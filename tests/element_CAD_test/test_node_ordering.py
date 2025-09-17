"""
Test the node ordering for second-order element conversion.

This script creates simple tetrahedron (C3D4) and wedge (C3D6) elements,
converts them to second-order elements (C3D10 and C3D15),
and visualizes the node numbering before and after conversion.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FEA.convert_elements import convert_to_second_order, convert_elements_batch
from FEA.generate_shell import generate_shell_from_surface, add_shell_elements_to_model
import FEA
from FEA.controller import FEAController
from FEA.elements.C3.C3D4 import C3D4
from FEA.elements.C3.C3D6 import C3D6
from FEA.elements.C3.C3D10 import C3D10
from FEA.elements.C3.C3D15 import C3D15
from FEA.convert_elements import convert_to_second_order

def generate_shell_model(fe: FEA.controller.FEAController, surface_names, shell_thickness: float) -> FEA.controller.FEAController:
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

def create_tetrahedron():
    """Create a simple tetrahedron model."""
    # Define nodes for a tetrahedron
    nodes = torch.tensor([
        [0.0, 0.0, 0.0],  # Node 0
        [1.0, 0.0, 0.0],  # Node 1
        [0.0, 1.0, 0.0],  # Node 2
        [0.0, 0.0, 1.0]   # Node 3
    ], dtype=torch.float32)
    
    # Create FEA model
    fe = FEAController(nodes)
    
    # Create a single tetrahedron element
    elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    elem_indices = torch.tensor([1], dtype=torch.int64)
    
    # Add the element to the model
    fe.add_element(C3D4(elems=elements, elems_index=elem_indices), name="tetrahedron")
    
    return fe

def create_wedge():
    """Create a simple wedge (prism) model."""
    # Define nodes for a wedge (triangular prism)
    nodes = torch.tensor([[-1.1231,  3.7917, 17.0000],
        [ 1.3201,  3.7277, 17.0000],
        [ 0.2929,  1.7165, 17.0000],
        [-0.5900,  2.5161, 15.5548],
        [ 0.9421,  2.1310, 15.8565],
        [ 0.2929,  1.7165, 15.0000]], dtype=torch.float32)
    
    # Create FEA model
    fe = FEAController(nodes)
    
    # Create a single wedge element
    elements = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int64)
    elem_indices = torch.tensor([1], dtype=torch.int64)
    
    # Add the element to the model
    fe.add_element(C3D6(elems=elements, elems_index=elem_indices), name="wedge")
    
    return fe

def visualize_element_nodes(fe, element_name, ax, title, ind=0):
    """Visualize element nodes with numbering."""
    # Get the element object and nodes
    elem_obj = fe.elems[element_name]
    nodes = fe.nodes.detach().cpu().numpy()
    
    # Get the first element's node indices
    if elem_obj._elems.shape[0] > 0:
        elem_nodes = elem_obj._elems[ind].detach().cpu().numpy()
    else:
        print(f"No elements found for {element_name}")
        return
    
    # Plot nodes
    ax.scatter(nodes[elem_nodes, 0], nodes[elem_nodes, 1], nodes[elem_nodes, 2], c='b', s=50)
    
    # Add node numbering
    for i in range(elem_obj._elems.shape[1]):
        ax.text(nodes[elem_nodes[i], 0], nodes[elem_nodes[i], 1], nodes[elem_nodes[i], 2], f"{i}", fontsize=12, color='black')
    
    # Plot element edges
    if isinstance(elem_obj, (C3D4, C3D10)):
        # For tetrahedron (all pairs of nodes)
        edges = []
        if isinstance(elem_obj, C3D4):
            # First-order tetrahedron edges
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        else:
            # Second-order tetrahedron edges
            # Include corner nodes connections
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            # Include mid-edge nodes connections
            edges.extend([(0, 4), (1, 4), (1, 5), (2, 5), (0, 6), (2, 6),
                         (0, 7), (3, 7), (1, 8), (3, 8), (2, 9), (3, 9)])
        
        for i, j in edges:
            if i < len(elem_nodes) and j < len(elem_nodes):
                node_i = nodes[elem_nodes[i]]
                node_j = nodes[elem_nodes[j]]
                ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], [node_i[2], node_j[2]], 'k-')
    
    elif isinstance(elem_obj, (C3D6, C3D15)):
        # For wedge
        edges = []
        if isinstance(elem_obj, C3D6):
            # First-order wedge edges
            edges = [
                # Bottom triangle
                (0, 1), (1, 2), (2, 0),
                # Top triangle
                (3, 4), (4, 5), (5, 3),
                # Vertical edges
                (0, 3), (1, 4), (2, 5)
            ]
        else:
            # Second-order wedge edges
            # Include corner nodes connections
            edges = [
                # Bottom triangle
                (0, 1), (1, 2), (2, 0),
                # Top triangle
                (3, 4), (4, 5), (5, 3),
                # Vertical edges
                (0, 3), (1, 4), (2, 5)
            ]
            # Include mid-edge nodes connections
            edges.extend([
                # Bottom triangle mid-edges
                (0, 6), (1, 6), (1, 7), (2, 7), (0, 8), (2, 8),
                # Top triangle mid-edges
                (3, 9), (4, 9), (4, 10), (5, 10), (3, 11), (5, 11),
                # Vertical mid-edges
                (0, 12), (3, 12), (1, 13), (4, 13), (2, 14), (5, 14)
            ])
        
        for i, j in edges:
            if i < len(elem_nodes) and j < len(elem_nodes):
                node_i = nodes[elem_nodes[i]]
                node_j = nodes[elem_nodes[j]]
                ax.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], [node_i[2], node_j[2]], 'k-')
    
    # Set the title and limits
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    max_range = np.array([
        nodes[:, 0].max() - nodes[:, 0].min(),
        nodes[:, 1].max() - nodes[:, 1].min(),
        nodes[:, 2].max() - nodes[:, 2].min()
    ]).max() * 0.5
    
    mid_x = (nodes[:, 0].max() + nodes[:, 0].min()) * 0.5
    mid_y = (nodes[:, 1].max() + nodes[:, 1].min()) * 0.5
    mid_z = (nodes[:, 2].max() + nodes[:, 2].min()) * 0.5
    
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add a legend
    if isinstance(elem_obj, (C3D4, C3D6)):
        ax.scatter([], [], c='b', s=50, label='Node')
    else:
        # Highlight mid-edge nodes differently
        corner_nodes = nodes[elem_nodes[:elem_obj.order + 1 if isinstance(elem_obj, C3D4) else 6]]
        ax.scatter(corner_nodes[:, 0], corner_nodes[:, 1], corner_nodes[:, 2], c='b', s=50, label='Corner Node')
        
        mid_nodes = nodes[elem_nodes[elem_obj.order + 1 if isinstance(elem_obj, C3D4) else 6:]]
        ax.scatter(mid_nodes[:, 0], mid_nodes[:, 1], mid_nodes[:, 2], c='r', s=50, label='Mid-Edge Node')
    
    ax.legend()

def test_node_ordering():
    fem = FEA.inp()
    current_path = os.path.dirname(os.path.abspath(__file__))
    inp_file = os.path.join(current_path, '_forshellgenerate.inp')
    fem.Read_INP(inp_file)
    
    # Initialize the FEA model
    fe = FEA.from_inp(fem)
    
    print("Original model stats:")
    print(f"Nodes: {fe.nodes.shape[0]}")
    print(f"Element groups: {len(fe.elems)}")
    for elem_name, elem_obj in fe.elems.items():
        print(f"  {elem_name}: {elem_obj._elems.shape[0]} elements")
      # Set shell thickness (can be adjusted as needed)
    shell_thickness = 2.5
    
    # Get all available surface names
    available_surfaces = [name for name in fe.surface_sets.keys() 
                         if (name.startswith('surface_') and name.endswith('_All') and name != 'surface_0_All')]  # Exclude exterior surface
    
    print(f"Available surfaces for shell generation: {available_surfaces}")
    

    
    # Generate new model with shell elements from multiple surfaces
    # You can specify a single surface: 'surface_1_All'
    # Or multiple surfaces: ['surface_1_All', 'surface_2_All', ...]
    new_fe = generate_shell_model(fe, available_surfaces, shell_thickness)
    
    new_fe_2nd = convert_to_second_order(new_fe)
    
    """Test the node ordering for second-order element conversion."""
    # # Create and convert tetrahedron
    # print("Creating tetrahedron model...")
    
    inp_file = os.path.join(current_path, 'C3D15.inp')
    fem.Read_INP(inp_file)
    
    # Initialize the FEA model
    fe = FEA.from_inp(fem)
    
    ind_tet = 0
    fe_tet = create_tetrahedron()
    # fe_tet.nodes = new_fe.nodes[new_fe.elems['element-0']._elems[ind_tet]]
    
    print("Converting tetrahedron to second-order...")
    fe_tet_2nd = convert_to_second_order(fe_tet)
    
    # Create and convert wedge
    print("Creating wedge model...")
    fe_wedge = create_wedge()
    ind_tet = -2
    # fe_wedge.nodes = new_fe.nodes[new_fe.elems['shell_elements']._elems[ind_tet]]
    
    print("Converting wedge to second-order...")
    fe_wedge_2nd = convert_to_second_order(fe_wedge)
    
    # Visualize both elements before and after conversion
    fig = plt.figure(figsize=(20, 10))
    
    # # Tetrahedron before conversion
    # ax1 = fig.add_subplot(221, projection='3d')
    # visualize_element_nodes(fe_tet, "tetrahedron", ax1, "Tetrahedron (C3D4)")
    
    # # Tetrahedron after conversion
    # ax2 = fig.add_subplot(222, projection='3d')
    # visualize_element_nodes(fe_tet_2nd, "tetrahedron", ax2, "Tetrahedron (C3D10)")
    
    # Tetrahedron before conversion
    ax1 = fig.add_subplot(221, projection='3d')
    visualize_element_nodes(fe, "element-0", ax1, "Tetrahedron (C3D15)")
    
    # Tetrahedron after conversion
    ax2 = fig.add_subplot(222, projection='3d')
    visualize_element_nodes(new_fe_2nd, "shell_elements", ax2, "Shell", ind=165)
    
    # Wedge before conversion
    ax3 = fig.add_subplot(223, projection='3d')
    visualize_element_nodes(fe_wedge, "wedge", ax3, "Wedge (C3D6)")
    
    # Wedge after conversion
    ax4 = fig.add_subplot(224, projection='3d')
    visualize_element_nodes(fe_wedge_2nd, "wedge", ax4, "Wedge (C3D15)")
    
    # Add overall title
    fig.suptitle('Node Ordering in Element Conversion', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    save_dir = os.path.join(os.path.dirname(__file__), 'node_ordering_visualization.png')
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_dir}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    test_node_ordering()
