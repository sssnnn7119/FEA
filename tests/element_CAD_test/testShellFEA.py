import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')

import FEA
from FEA.elements import materials

os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

fem = FEA.FEA_INP()

from matplotlib import pyplot as plt

def check_for_duplicate_nodes(fe, tol=1e-6, visualize=False):
    """
    Check if there are any duplicate nodes in the FE model with enhanced reporting.
    
    Args:
        fe (FEA.Main.FEA_Main): The FE model to check
        tol (float, optional): Tolerance for considering nodes as duplicates. Defaults to 1e-6.
        visualize (bool, optional): Whether to create a visualization of duplicate nodes. Defaults to False.
        
    Returns:
        tuple: (has_duplicates, duplicate_info)
            - has_duplicates (bool): True if duplicate nodes were found
            - duplicate_info (dict): Dictionary with detailed information about duplicates
    """
    # Extract all node coordinates
    nodes = fe.nodes.cpu()
    
    # Keep track of nodes that are very close to each other
    duplicate_count = 0
    duplicate_pairs = []
    duplicate_info = {
        "count": 0,
        "pairs": [],
        "clusters": [],
        "distances": []
    }
    
    # Build clusters of duplicate nodes
    clusters = []
    processed_nodes = set()
    
    # For each node, check if there's another node very close to it
    for i in range(nodes.shape[0]):
        if i in processed_nodes:
            continue
            
        # Calculate distances from this node to all other nodes
        diffs = nodes - nodes[i]
        distances = torch.sqrt(torch.sum(diffs * diffs, dim=1))
        
        # Find nodes that are very close but not the same node
        close_mask = (distances > 0) & (distances < tol)
        
        if torch.any(close_mask):
            # This node has duplicates - create a cluster
            close_indices = torch.where(close_mask)[0].tolist()
            
            # Create a new cluster with this node and its duplicates
            cluster = [i] + close_indices
            clusters.append(cluster)
            
            # Mark all these nodes as processed
            for node_idx in cluster:
                processed_nodes.add(node_idx)
                
            # Record the duplicate pairs and their distances
            for idx in close_indices:
                duplicate_pairs.append((i, idx))
                duplicate_info["pairs"].append((i, idx))
                duplicate_info["distances"].append(distances[idx].item())
                print(f"Duplicate nodes found: {i} and {idx}")
                print(f"  Coordinates 1: {nodes[i]}")
                print(f"  Coordinates 2: {nodes[idx]}")
                print(f"  Distance: {distances[idx].item()}")
            
            duplicate_count += len(close_indices)
    
    duplicate_info["count"] = duplicate_count
    duplicate_info["clusters"] = clusters
    
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate node pairs in {len(clusters)} clusters")
        
        if visualize and duplicate_count > 0:
            visualize_duplicate_nodes(fe, clusters)
            
        return True, duplicate_info
    else:
        print("No duplicate nodes found")
        return False, duplicate_info


def visualize_duplicate_nodes(fe, clusters):
    """
    Visualize duplicate nodes in a 3D plot.
    
    Args:
        fe (FEA.Main.FEA_Main): The FE model with nodes to visualize
        clusters (list): List of clusters of duplicate node indices
    """
    # Handle case where there are no clusters
    if not clusters:
        print("No duplicate nodes to visualize")
        return None
        
    nodes = fe.nodes.cpu().numpy()
    
    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all nodes in light gray
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], 
               color='lightgray', alpha=0.3, s=10)
    
    # Plot each cluster of duplicate nodes with a distinct color
    colors = plt.cm.tab10.colors
    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]
        cluster_nodes = nodes[cluster]
        
        # Plot the duplicate nodes
        ax.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1], cluster_nodes[:, 2],
                   color=color, s=50, label=f"Cluster {i+1}")
        
        # Connect the nodes in this cluster with lines
        for j in range(len(cluster)):
            for k in range(j+1, len(cluster)):
                ax.plot([cluster_nodes[j, 0], cluster_nodes[k, 0]],
                        [cluster_nodes[j, 1], cluster_nodes[k, 1]],
                        [cluster_nodes[j, 2], cluster_nodes[k, 2]],
                        color=color, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Duplicate Node Visualization')
    
    # Create a legend with only one entry per cluster
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def analyze_element_connectivity(fe, duplicate_info):
    """
    Analyze which elements are using the duplicate nodes.
    
    Args:
        fe (FEA.Main.FEA_Main): The FE model to check
        duplicate_info (dict): Dictionary with duplicate node information
        
    Returns:
        dict: Dictionary mapping node index to list of (element_name, element_index) tuples
    """
    node_to_elements = {}
    
    # Initialize the dictionary for all nodes involved in duplicates
    for cluster in duplicate_info["clusters"]:
        for node_idx in cluster:
            node_to_elements[node_idx] = []
    
    # Scan through all elements to find which ones use the duplicate nodes
    for elem_name, elem_obj in fe.elems.items():
        elem_connectivity = elem_obj._elems
        
        # For each element in this set
        for i in range(elem_connectivity.shape[0]):
            elem_nodes = elem_connectivity[i]
            
            # Check if any of the element's nodes are in our duplicate nodes list
            for node_idx in node_to_elements.keys():
                if node_idx in elem_nodes:
                    node_to_elements[node_idx].append((elem_name, i))
    
    return node_to_elements


def fix_duplicate_nodes(fe, duplicate_info, tolerance=1e-6):
    """
    Fix duplicate nodes in the FE model by merging them.
    
    Args:
        fe (FEA.Main.FEA_Main): The FE model to fix
        duplicate_info (dict): Dictionary with duplicate node information
        tolerance (float, optional): Tolerance for considering nodes as duplicates. Defaults to 1e-6.
        
    Returns:
        tuple: (fixed_fe, merged_nodes)
            - fixed_fe (FEA.Main.FEA_Main): FE model with merged nodes
            - merged_nodes (dict): Dictionary mapping merged node indices to their new indices
    """
    # This is a placeholder for future implementation
    # Implementation would:
    # 1. Create a mapping from old node indices to new node indices
    # 2. Create a new nodes tensor with duplicates removed
    # 3. Update all element connectivity to use the new node indices
    # 4. Create a new FE model with the updated nodes and elements
    
    print("Node merging functionality not yet implemented")
    return fe, {}


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
    nodes_new, c3d6_elements, c3d6_indices, offset_surface_sets = FEA.elements.generate_shell_from_surface(
        fe, surface_names, shell_thickness)
    
    # Add the shell elements to create a new model
    new_fe = FEA.elements.add_shell_elements_to_model(fe, nodes_new, c3d6_elements, c3d6_indices, offset_surface_sets= offset_surface_sets)
    
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
    second_order_fe = FEA.elements.convert_elements_batch(fe, element_names_to_convert)
    
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
    shell_thickness = 1.5
    
    # Get all available surface names
    available_surfaces = [name for name in fe.surface_sets.keys() 
                         if (name.startswith('surface_') and name.endswith('_All') and name!='surface_0_All')]  # Exclude exterior surface
    
    print(f"Available surfaces for shell generation: {available_surfaces}")
    
    # Generate new model with shell elements from multiple surfaces
    # You can specify a single surface: 'surface_1_All'
    # Or multiple surfaces: ['surface_1_All', 'surface_2_All', ...]
    fe = generate_shell_model(fe, available_surfaces, shell_thickness)
    
    print("\nNew model with shell elements:")
    print(f"Nodes: {fe.nodes.shape[0]}")
    print(f"Element groups: {len(fe.elems)}")
    for elem_name, elem_obj in fe.elems.items():
        print(f"  {elem_name}: {elem_obj._elems.shape[0] if hasattr(elem_obj, '_elems') else 0} elements")
      # Convert both original and shell elements to second-order elements
    print("\nConverting elements to second-order...")
    # Convert all element sets to second-order (C3D4->C3D10, C3D6->C3D15)
    element_names = list(fe.elems.keys())
    print(f"Element sets to convert: {element_names}")
    
    # Use the batch conversion function to convert all elements
    fe = FEA.elements.convert_to_second_order(fe, element_names)
    
    fe.elems['element-0'].set_materials(FEA.materials.hyperelastic.NeoHookean(
        mu=torch.tensor(0.001, device=fe.nodes.device),
        kappa=torch.tensor(0.002, device=fe.nodes.device)))
    fe.elems['shell_elements'].set_materials(FEA.materials.hyperelastic.NeoHookean(
        mu=torch.tensor(0.48, device=fe.nodes.device),
        kappa=torch.tensor(4.8, device=fe.nodes.device)))
    
    fe.add_load(FEA.loads.Pressure(surface_set='surface_1_All_offset', pressure=0.02),
                    name='pressure-1')

    bc_dof = np.where((abs(fe.nodes[:, 2] - 0)
                            < 0.1).cpu().numpy())[0] * 3
    bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
    bc_name = fe.add_constraint(
        FEA.constraints.Boundary_Condition(indexDOF=bc_dof,
                                        dispValue=torch.zeros(bc_dof.size)))

    rp = fe.add_reference_point(FEA.ReferencePoint([0, 0, 20]))

    indexNodes = np.where((abs(fe.nodes[:, 2] - 20)
                            < 0.1).cpu().numpy())[0]
    fe.initialize()
    fe.export_to_inp('Z:/cache/testShellFEA.inp')

    fe.add_constraint(
        FEA.constraints.Couple(
            indexNodes=indexNodes, rp_name=rp))
    
    fe.elems['element-0'].set_order(1)
    fe.elems['shell_elements'].set_order(1)
    fe.solve(tol_error=0.001)
    
    print(fe.GC[-6:])

