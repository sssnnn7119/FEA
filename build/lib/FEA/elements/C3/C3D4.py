import numpy as np
import torch
from .C3base import Element_3D
from .C3D10 import C3D10

class C3D4(Element_3D):
    """
        Local coordinates:
            origin: 0-th nodal
            \ksi_0: 0-1 vector
            \ksi_1: 0-2 vector
            \ksi_2: 0-3 vector

        face nodal always point at the void
            face0: 021
            face1: 013
            face2: 123
            face3: 032

        shape_funtion:
            N_i = \ksi_i * \ksi_i, i<=3
    """

    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)

        self.order = 1
        
    

    def initialize(self, fea):
        
        self.shape_function = [
            torch.tensor([[1.0, -1.0, -1.0, -1.0], [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                          [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                          [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
        ]

        self.num_nodes_per_elem = 4
        self._num_gaussian = 1
        self.gaussian_weight = torch.tensor([1 / 6])

        p0 = torch.tensor([[0.25, 0.25, 0.25]])
        self._pre_load_gaussian(p0, nodes=fea.nodes)
        super().initialize(fea)
        
    
    def find_surface(self, surface_ind: int,
                           elems_ind: torch.Tensor):

        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]

        if surface_ind == 0:
            return self._elems[index_now][:, [0, 2, 1]]
        elif surface_ind == 1:
            return self._elems[index_now][:, [0, 1, 3]]
        elif surface_ind == 2:
            return self._elems[index_now][:, [1, 2, 3]]
        elif surface_ind == 3:
            return self._elems[index_now][:, [0, 3, 2]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")

    def to_C3D10(self, nodes_now: torch.Tensor):
        """
        Convert C3D4 element to C3D10 element.
        Returns:
            C3D10: C3D10 element with the same nodes as this C3D4 element.
        """

        elems_now = self._elems.clone()
        
        # Create quadratic elements with mid-edge nodes
        # Node order: 4(01) 5(12) 6(02) 7(03) 8(13) 9(23)
        # Where 4 is the midpoint of nodes 0 and 1, etc.
        
        # Define edge pairs for tetrahedral elements
        edge_pairs = torch.tensor([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]], device=elems_now.device)
        num_elems = elems_now.shape[0]
        num_edges_per_elem = edge_pairs.shape[0]
        
        # Step 1: Extract all edges from all elements (vectorized)
        # Get the node indices for each edge of each element
        edges_i = elems_now[:, edge_pairs[:, 0]].reshape(-1)  # First node of each edge
        edges_j = elems_now[:, edge_pairs[:, 1]].reshape(-1)  # Second node of each edge
        
        # Ensure i < j for consistent ordering
        min_idx = torch.minimum(edges_i, edges_j)
        max_idx = torch.maximum(edges_i, edges_j)
        edges_i, edges_j = min_idx, max_idx
        
        # Combine indices into a 2D tensor for unique edge detection
        edges_tensor = torch.stack([edges_i, edges_j], dim=1)
        
        # Find unique edges
        unique_edges, inverse_indices = torch.unique(edges_tensor, dim=0, return_inverse=True)
        
        # Create midpoint nodes for unique edges
        mid_nodes = (nodes_now[unique_edges[:, 0]] + nodes_now[unique_edges[:, 1]]) / 2.0
        
        # Add new nodes to existing nodes
        nodes_new = torch.cat([nodes_now, mid_nodes], dim=0)
        
        # Create a mapping from edge indices to new node indices
        edge_to_node_idx = torch.arange(nodes_now.shape[0], nodes_new.shape[0], device=elems_now.device)
        
        # Reshape inverse_indices to map back to the original element-edge pairs
        elem_edge_indices = inverse_indices.reshape(num_elems, num_edges_per_elem)
        
        # Step 2: Create new elements with proper node ordering (vectorized)
        # First 4 nodes are the original nodes
        elems_first_four = elems_now.clone()
        
        # Calculate indices for mid-edge nodes in the order specified: 4(01), 5(12), 6(02), 7(03), 8(13), 9(23)
        # Map the edge indices to the new node indices
        mid_edge_indices = nodes_now.shape[0] + elem_edge_indices
        
        # Create new 10-node elements
        elems_new = torch.cat([
            elems_first_four,  # Original 4 nodes
            mid_edge_indices   # 6 mid-edge nodes
        ], dim=1)
        
        C3D10_element = C3D10(elems=elems_new, elems_index=self._elems_index)
        C3D10_element.set_materials(self.materials)
        C3D10_element.set_density(self.density)
        
        return nodes_new, C3D10_element