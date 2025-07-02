import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
