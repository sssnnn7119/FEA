import numpy as np
import torch
from .C3base import Element_3D

class C3D20(Element_3D):
    """
    C3D20 - 20-node quadratic brick element
    
    # Local coordinates:
        origin: center of brick element
        g, h, r: local coordinates aligned with element edges
        All coordinates vary from -1 to 1
    
    # Node numbering follows Abaqus convention:
        Bottom face (r=-1):
            0: (-1, -1, -1) - corner
            1: ( 1, -1, -1) - corner
            2: ( 1,  1, -1) - corner
            3: (-1,  1, -1) - corner
            8: ( 0, -1, -1) - mid-edge
            9: ( 1,  0, -1) - mid-edge
           10: ( 0,  1, -1) - mid-edge
           11: (-1,  0, -1) - mid-edge
        Top face (r=1):
            4: (-1, -1,  1) - corner
            5: ( 1, -1,  1) - corner
            6: ( 1,  1,  1) - corner
            7: (-1,  1,  1) - corner
           12: ( 0, -1,  1) - mid-edge
           13: ( 1,  0,  1) - mid-edge
           14: ( 0,  1,  1) - mid-edge
           15: (-1,  0,  1) - mid-edge
        Middle edges (r=0):
           16: (-1, -1,  0) - mid-edge
           17: ( 1, -1,  0) - mid-edge
           18: ( 1,  1,  0) - mid-edge
           19: (-1,  1,  0) - mid-edge
            
    # Face definitions:
        face0: 0-3-2-1 (nodes 0,3,2,1,11,10,9,8) (Bottom face, r=-1)
        face1: 4-5-6-7 (nodes 4,5,6,7,12,13,14,15) (Top face, r=1)
        face2: 0-4-7-3 (nodes 0,4,7,3,16,15,19,11) (Left face, g=-1)
        face3: 1-2-6-5 (nodes 1,2,6,5,9,18,13,17) (Right face, g=1)
        face4: 0-1-5-4 (nodes 0,1,5,4,8,17,12,16) (Front face, h=-1)
        face5: 3-7-6-2 (nodes 3,7,6,2,19,14,18,10) (Back face, h=1)    # Shape functions:
        Quadratic serendipity shape functions for brick element
        N_i = combination of 1, g, h, r, g^2, h^2, r^2, gh, hr, rg, ghr
        Corner nodes use the product of quadratic terms
        Mid-edge nodes use specific quadratic functions
    """
    
    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)
        self.order = 2
    
    def initialize(self, fea):
        # Shape function coefficients for 20-node brick element with coordinates (g,h,r)
        # According to the provided function ordering in C3base.py:
        # 0: constant, 1: g, 2: h, 3: r, 4: g*h, 5: h*r, 6: r*g, 
        # 7: g^2, 8: h^2, 9: r^2, 10: g^2*h, 11: g*h^2, 12: h^2*r, 
        # 13: h*r^2, 14: r^2*g, 15: r*g^2, 16: g*h*r, 17: g^2*h*r, 
        # 18: g*h^2*r, 19: g*h*r^2
          # Initialize shape functions for all nodes using standard C3D20 formulas
        shape_funcs = torch.zeros((20, 20))  # 20 nodes, 20 coefficients (max)
        derivatives = torch.zeros((3, 20, 20))  # 3 derivatives (g,h,r), 20 nodes
        
        # Standard C3D20 shape functions in matrix form
        # Based on the formulas provided:
        # Corner nodes: N_i = -(1/8)(1±g)(1±h)(1±r)(2±g±h±r)
        # Mid-edge nodes: N_i = (1/4)(1-g²)(1±h)(1±r) or similar for other directions
        
        # Define the shape function coefficients directly as a matrix
        # Each row corresponds to one shape function, each column to a polynomial term
        # 0: const, 1: g, 2: h, 3: r, 4: gh, 5: hr, 6: rg, 7: g², 8: h², 9: r²,
        # 10: g²h, 11: gh², 12: h²r, 13: hr², 14: r²g, 15: rg², 16: ghr, 17: g²hr, 18: gh²r, 19: ghr²
        
        # Initialize the coefficient matrix directly
        shape_funcs = torch.tensor([
            [ -0.2500,   0.1250,   0.1250,   0.1250,   0.0000,   0.0000,   0.0000,   0.1250,   0.1250,   0.1250,  -0.1250,  -0.1250,  -0.1250,  -0.1250,  -0.1250,  -0.1250,  -0.1250,   0.0000,   0.0000,   0.0000],  # 节点 1
            [ -0.2500,  -0.1250,   0.1250,   0.1250,   0.0000,   0.0000,   0.0000,   0.1250,   0.1250,   0.1250,  -0.1250,   0.1250,  -0.1250,  -0.1250,   0.1250,  -0.1250,   0.1250,   0.0000,   0.0000,   0.0000],  # 节点 2
            [ -0.2500,  -0.1250,  -0.1250,   0.1250,   0.0000,   0.0000,   0.0000,   0.1250,   0.1250,   0.1250,   0.1250,   0.1250,  -0.1250,   0.1250,   0.1250,  -0.1250,  -0.1250,   0.0000,   0.0000,   0.0000],  # 节点 3
            [ -0.2500,   0.1250,  -0.1250,   0.1250,   0.0000,   0.0000,   0.0000,   0.1250,   0.1250,   0.1250,   0.1250,  -0.1250,  -0.1250,   0.1250,  -0.1250,  -0.1250,   0.1250,   0.0000,   0.0000,   0.0000],  # 节点 4
            [ -0.2500,   0.1250,   0.1250,  -0.1250,   0.0000,   0.0000,   0.0000,   0.1250,   0.1250,   0.1250,  -0.1250,  -0.1250,   0.1250,  -0.1250,  -0.1250,   0.1250,   0.1250,   0.0000,   0.0000,   0.0000],  # 节点 5
            [ -0.2500,  -0.1250,   0.1250,  -0.1250,   0.0000,   0.0000,   0.0000,   0.1250,   0.1250,   0.1250,  -0.1250,   0.1250,   0.1250,  -0.1250,   0.1250,   0.1250,  -0.1250,   0.0000,   0.0000,   0.0000],  # 节点 6
            [ -0.2500,  -0.1250,  -0.1250,  -0.1250,   0.0000,   0.0000,   0.0000,   0.1250,   0.1250,   0.1250,   0.1250,   0.1250,   0.1250,   0.1250,   0.1250,   0.1250,   0.1250,   0.0000,   0.0000,   0.0000],  # 节点 7
            [ -0.2500,   0.1250,  -0.1250,  -0.1250,   0.0000,   0.0000,   0.0000,   0.1250,   0.1250,   0.1250,   0.1250,  -0.1250,   0.1250,   0.1250,  -0.1250,   0.1250,  -0.1250,   0.0000,   0.0000,   0.0000],  # 节点 8
            [  0.2500,   0.0000,  -0.2500,  -0.2500,   0.0000,   0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 9
            [  0.2500,   0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,  -0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,  -0.2500,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 10
            [  0.2500,   0.0000,   0.2500,  -0.2500,   0.0000,  -0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 11
            [  0.2500,  -0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,   0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,   0.2500,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 12
            [  0.2500,   0.0000,  -0.2500,   0.2500,   0.0000,  -0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 13
            [  0.2500,   0.2500,   0.0000,   0.2500,   0.0000,   0.0000,   0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,  -0.2500,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 14
            [  0.2500,   0.0000,   0.2500,   0.2500,   0.0000,   0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 15
            [  0.2500,  -0.2500,   0.0000,   0.2500,   0.0000,   0.0000,  -0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,   0.2500,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 16
            [  0.2500,  -0.2500,  -0.2500,   0.0000,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,   0.2500,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 17
            [  0.2500,   0.2500,  -0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,   0.2500,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 18
            [  0.2500,   0.2500,   0.2500,   0.0000,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,  -0.2500,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000],  # 节点 19
            [  0.2500,  -0.2500,   0.2500,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,   0.0000,  -0.2500,   0.0000,   0.0000,   0.0000,  -0.2500,   0.2500,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000]  # 节点 20
        ])
        
        # evaluate derivatives for each shape function
        for i in range(3):
            derivatives[i] = self._shape_function_derivative(shape_funcs, i)
        
        # Store shape function coefficients
        self.shape_function = [shape_funcs, derivatives]
        self.num_nodes_per_elem = 20
        
        # Gaussian integration points
        # Using 3x3x3 Gauss quadrature for full integration
        self._num_gaussian = 27  # 3x3x3 points
        
        # Gaussian weights (3x3x3)
        weight_1d = torch.tensor([5.0/9.0, 8.0/9.0, 5.0/9.0])
        weights = torch.zeros(27)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weights[idx] = weight_1d[i] * weight_1d[j] * weight_1d[k]
                    idx += 1
        self.gaussian_weight = weights
        
        # Gaussian points in natural coordinates
        points_1d = torch.tensor([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        p0 = torch.zeros((27, 3))
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    p0[idx, 0] = points_1d[i]  # g
                    p0[idx, 1] = points_1d[j]  # h
                    p0[idx, 2] = points_1d[k]  # r
                    idx += 1
        
        # Load Gaussian points for integration
        self._pre_load_gaussian(p0, nodes=fea.nodes)
        super().initialize(fea)
    
    def find_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        """
        Find surface elements for this element type
        
        Args:
            surface_ind: Surface index (0-5)
            elems_ind: Element indices to find surfaces for
            
        Returns:
            Tensor with surface node indices
        """
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]
        
        if index_now.shape[0] == 0:
            return torch.empty([0, 8], dtype=torch.long, device=self._elems.device)
        
        if surface_ind == 0:  # Bottom face (r=-1): 0,3,2,1 + mid nodes 11,10,9,8
            return self._elems[index_now][:, [0, 3, 2, 1, 11, 10, 9, 8]]
        elif surface_ind == 1:  # Top face (r=1): 4,5,6,7 + mid nodes 12,13,14,15
            return self._elems[index_now][:, [4, 5, 6, 7, 12, 13, 14, 15]]
        elif surface_ind == 2:  # Left face (g=-1): 0,4,7,3 + mid nodes 16,15,19,11
            return self._elems[index_now][:, [0, 4, 7, 3, 16, 15, 19, 11]]
        elif surface_ind == 3:  # Right face (g=1): 1,2,6,5 + mid nodes 9,18,13,17
            return self._elems[index_now][:, [1, 2, 6, 5, 9, 18, 13, 17]]
        elif surface_ind == 4:  # Front face (h=-1): 0,1,5,4 + mid nodes 8,17,12,16
            return self._elems[index_now][:, [0, 1, 5, 4, 8, 17, 12, 16]]
        elif surface_ind == 5:  # Back face (h=1): 3,7,6,2 + mid nodes 19,14,18,10
            return self._elems[index_now][:, [3, 7, 6, 2, 19, 14, 18, 10]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
    
    def refine_RGC(self, RGC: list[torch.Tensor], nodes: torch.Tensor) -> list[torch.Tensor]:
        """
        Refine Reference Grid Coordinates for mid-edge nodes
        
        Args:
            RGC: List of Reference Grid Coordinates
            nodes: Node coordinates
            
        Returns:
            Updated RGC
        """
        mid_nodes_index = self.get_2nd_order_point_index()
        RGC[0][mid_nodes_index[:, 0]] = (RGC[0][mid_nodes_index[:, 1]] + RGC[0][mid_nodes_index[:, 2]]) / 2 + (nodes[mid_nodes_index[:, 1]] + nodes[mid_nodes_index[:, 2]] - 2 * nodes[mid_nodes_index[:, 0]]) / 2
        
        return RGC
    
    def set_order(self, order: int) -> None:
        """
        Set the order of the element
        
        Args:
            order: Element order (1 for linear, 2 for quadratic)
        """
        self.order = order
    
    def get_2nd_order_point_index(self):
        """
        Get mid-edge node indices with their corner node neighbors
        
        Returns:
            Tensor with [mid_node_idx, corner_node1_idx, corner_node2_idx]
        """
        # 12 edges for hex element: (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), 
        # (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)
        mid_index = torch.cat([
            self._elems[:, 8],  # Edge (0,1)
            self._elems[:, 9],  # Edge (1,2)
            self._elems[:, 10], # Edge (2,3)
            self._elems[:, 11], # Edge (3,0)
            self._elems[:, 12], # Edge (4,5)
            self._elems[:, 13], # Edge (5,6)
            self._elems[:, 14], # Edge (6,7)
            self._elems[:, 15], # Edge (7,4)
            self._elems[:, 16], # Edge (0,4)
            self._elems[:, 17], # Edge (1,5)
            self._elems[:, 18], # Edge (2,6)
            self._elems[:, 19]  # Edge (3,7)
        ])
        
        neighbor1_index = torch.cat([
            self._elems[:, 0],  # Node 0 (with 8)
            self._elems[:, 1],  # Node 1 (with 9)
            self._elems[:, 2],  # Node 2 (with 10)
            self._elems[:, 3],  # Node 3 (with 11)
            self._elems[:, 4],  # Node 4 (with 12)
            self._elems[:, 5],  # Node 5 (with 13)
            self._elems[:, 6],  # Node 6 (with 14)
            self._elems[:, 7],  # Node 7 (with 15)
            self._elems[:, 0],  # Node 0 (with 16)
            self._elems[:, 1],  # Node 1 (with 17)
            self._elems[:, 2],  # Node 2 (with 18)
            self._elems[:, 3]   # Node 3 (with 19)
        ])
        
        neighbor2_index = torch.cat([
            self._elems[:, 1],  # Node 1 (with 8)
            self._elems[:, 2],  # Node 2 (with 9)
            self._elems[:, 3],  # Node 3 (with 10)
            self._elems[:, 0],  # Node 0 (with 11)
            self._elems[:, 5],  # Node 5 (with 12)
            self._elems[:, 6],  # Node 6 (with 13)
            self._elems[:, 7],  # Node 7 (with 14)
            self._elems[:, 4],  # Node 4 (with 15)
            self._elems[:, 4],  # Node 4 (with 16)
            self._elems[:, 5],  # Node 5 (with 17)
            self._elems[:, 6],  # Node 6 (with 18)
            self._elems[:, 7]   # Node 7 (with 19)
        ])
        
        # Sort and get unique nodes
        arg_index = torch.argsort(mid_index)
        result = torch.stack([mid_index, neighbor1_index, neighbor2_index], dim=1)
        result = result[arg_index]
        index_remain = torch.zeros([result.shape[0]], dtype=torch.bool, device='cpu')
        index_remain[0] = True
        index_remain[1:][result[1:, 0] > result[:-1, 0]] = True
        result = result[index_remain]
        
        return result
    
    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Set required degrees of freedom
        
        Args:
            RGC_remain_index: List of indices to keep
            
        Returns:
            Updated RGC_remain_index
        """
        # Always keep corner nodes
        RGC_remain_index[0][self._elems[:, :8].unique()] = True
        
        # Keep mid-nodes for quadratic elements
        if self.order > 1:
            RGC_remain_index[0][self._elems[:, 8:].unique()] = True
            
        return RGC_remain_index
