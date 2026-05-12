import numpy as np
import torch
from .C3base import Element_3D
from .surfaces import initialize_surfaces

# =============================================================================
# C3D8 : 8-node linear brick, full integration (2×2×2)
# =============================================================================
class C3D8(Element_3D):
    """
    C3D8 - 8-node linear brick, full integration
    
    Local coordinates: g, h, r ∈ [-1, 1]
        origin: element center
    
    Node numbering (Abaqus convention):
        Bottom face (r=-1):  0(-1,-1,-1)  1( 1,-1,-1)  2( 1, 1,-1)  3(-1, 1,-1)
        Top face    (r= 1):  4(-1,-1, 1)  5( 1,-1, 1)  6( 1, 1, 1)  7(-1, 1, 1)
            
    Face definitions:
        face0: 0321 (Bottom, r=-1)    face1: 4567 (Top, r=1)
        face2: 0154 (Left,  g=-1)    face3: 1265 (Right, g=1)
        face4: 2376 (Front, h=-1)    face5: 0473 (Back, h=1)

    Shape functions:
        N_i = 1/8 (1 + g·g_i)(1 + h·h_i)(1 + r·r_i)
    """

    # ---- class-level static attributes (same pattern as tetrahedral.py / wedge.py) ----

    # Trilinear shape function coefficients in polynomial basis:
    #   [1, g, h, r, g*h, h*r, r*g, ..., g*h*r]
    shape_function = [
        torch.tensor([
            [ 0.125, -0.125, -0.125, -0.125,  0.125,  0.125,  0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0.125,  0.,  0.,  0.],
            [ 0.125,  0.125, -0.125, -0.125, -0.125,  0.125, -0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.125,  0.,  0.,  0.],
            [ 0.125,  0.125,  0.125, -0.125,  0.125, -0.125, -0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0.125,  0.,  0.,  0.],
            [ 0.125, -0.125,  0.125, -0.125, -0.125, -0.125,  0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.125,  0.,  0.,  0.],
            [ 0.125, -0.125, -0.125,  0.125,  0.125, -0.125, -0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.125,  0.,  0.,  0.],
            [ 0.125,  0.125, -0.125,  0.125, -0.125, -0.125,  0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0.125,  0.,  0.,  0.],
            [ 0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.125,  0.,  0.,  0.],
            [ 0.125, -0.125,  0.125,  0.125, -0.125,  0.125, -0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0.125,  0.,  0.,  0.],
        ]),
    ]

    num_nodes_per_elem = 8
    num_surfaces = 6
    _num_gaussian = 8

    # Gauss-Legendre 2×2×2: weights = 1, points = ±1/√3
    gaussian_weight_ref = torch.ones(8)
    _p = 1.0 / np.sqrt(3.0)
    gaussian_coordinates = torch.tensor([
        [-_p, -_p, -_p],
        [ _p, -_p, -_p],
        [ _p,  _p, -_p],
        [-_p,  _p, -_p],
        [-_p, -_p,  _p],
        [ _p, -_p,  _p],
        [ _p,  _p,  _p],
        [-_p,  _p,  _p],
    ])

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        """
        Find the surface elements for a given surface index and element indices.
        
        Args:
            surface_ind: Surface index (0-5)
            elems_ind: Element indices
            
        Returns:
            torch.Tensor: Surface element node indices
        """
        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]

        if index_now.shape[0] == 0:
            quad_elems = torch.empty([0, 4],
                               dtype=torch.long,
                               device=self._elems.device)
            return [initialize_surfaces(quad_elems)]

        # Return appropriate face nodes according to face definitions in comments
        if surface_ind == 0:  # Bottom face (r=-1): face0: 0321
            quad_elems = self._elems[index_now][:, [0, 3, 2, 1]]
        elif surface_ind == 1:  # Top face (r=1): face1: 4567
            quad_elems = self._elems[index_now][:, [4, 5, 6, 7]]
        elif surface_ind == 2:  # Left face (g=-1): face2: 0154
            quad_elems = self._elems[index_now][:, [0, 1, 5, 4]]
        elif surface_ind == 3:  # Right face (g=1): face3: 1265
            quad_elems = self._elems[index_now][:, [1, 2, 6, 5]]
        elif surface_ind == 4:  # Front face (h=-1): face4: 2376
            quad_elems = self._elems[index_now][:, [2, 3, 7, 6]]
        elif surface_ind == 5:  # Back face (h=1): face5: 0473
            quad_elems = self._elems[index_now][:, [0, 4, 7, 3]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")

        return [initialize_surfaces(quad_elems)]


# =============================================================================
# C3D20 : 20-node quadratic serendipity brick, full integration (3×3×3)
# =============================================================================
class C3D20(Element_3D):
    """
    C3D20 - 20-node quadratic brick element (serendipity)
    
    Local coordinates: g, h, r ∈ [-1, 1], origin at element center.
    
    Node numbering (Abaqus convention):
        Bottom face (r=-1):  0-3-2-1  (corners),  11,10,9,8 (mid-edge)
        Top face    (r= 1):  4-5-6-7  (corners),  12,13,14,15 (mid-edge)
        Middle r=0  edges:   16(-1,-1,0)  17(1,-1,0)  18(1,1,0)  19(-1,1,0)
    
    Face definitions:
        face0: 0,3,2,1,11,10,9,8   (Bottom, r=-1)
        face1: 4,5,6,7,12,13,14,15 (Top, r=1)
        face2: 0,1,5,4,8,17,12,16  (Front, h=-1)
        face3: 1,2,6,5,9,18,13,17  (Right, g=1)
        face4: 2,3,7,6,10,19,14,18 (Back, h=1)
        face5: 0,4,7,3,16,15,19,11 (Left, g=-1)
    """

    # ---- class-level static attributes ----
    # Quadratic serendipity shape function coefficients (20 nodes × 20 basis terms)
    # Basis: [1, g, h, r, gh, hr, rg, g², h², r², g²h, gh², h²r, hr², r²g, rg², ghr, g²hr, gh²r, ghr²]
    shape_function = [
        torch.tensor([
            [-0.25,  0.125,  0.125,  0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125,  0.,     0.,     0.   ],
            [-0.25, -0.125,  0.125,  0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125, -0.125,  0.125, -0.125, -0.125,  0.125, -0.125,  0.125,  0.,     0.,     0.   ],
            [-0.25, -0.125, -0.125,  0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125,  0.125,  0.125, -0.125,  0.125,  0.125, -0.125, -0.125,  0.,     0.,     0.   ],
            [-0.25,  0.125, -0.125,  0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125,  0.125, -0.125, -0.125,  0.125, -0.125, -0.125,  0.125,  0.,     0.,     0.   ],
            [-0.25,  0.125,  0.125, -0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125, -0.125, -0.125,  0.125, -0.125, -0.125,  0.125,  0.125,  0.,     0.,     0.   ],
            [-0.25, -0.125,  0.125, -0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125, -0.125,  0.125,  0.125, -0.125,  0.125,  0.125, -0.125,  0.,     0.,     0.   ],
            [-0.25, -0.125, -0.125, -0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.,     0.,     0.   ],
            [-0.25,  0.125, -0.125, -0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125,  0.125, -0.125,  0.125,  0.125, -0.125,  0.125, -0.125,  0.,     0.,     0.   ],
            [ 0.25,  0.,    -0.25,  -0.25,   0.,     0.25,   0.,    -0.25,   0.,     0.,     0.25,   0.,     0.,     0.,     0.,     0.25,   0.,     0.,     0.,     0.   ],
            [ 0.25,  0.25,   0.,    -0.25,   0.,     0.,    -0.25,   0.,    -0.25,   0.,     0.,    -0.25,   0.25,   0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.,     0.25,  -0.25,   0.,    -0.25,   0.,    -0.25,   0.,     0.,    -0.25,   0.,     0.,     0.,     0.,     0.25,   0.,     0.,     0.,     0.   ],
            [ 0.25, -0.25,   0.,    -0.25,   0.,     0.,     0.25,   0.,    -0.25,   0.,     0.,     0.25,   0.25,   0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.,    -0.25,   0.25,   0.,    -0.25,   0.,    -0.25,   0.,     0.,     0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,     0.   ],
            [ 0.25,  0.25,   0.,     0.25,   0.,     0.,     0.25,   0.,    -0.25,   0.,     0.,    -0.25,  -0.25,   0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.,     0.25,   0.25,   0.,     0.25,   0.,    -0.25,   0.,     0.,    -0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,     0.   ],
            [ 0.25, -0.25,   0.,     0.25,   0.,     0.,    -0.25,   0.,    -0.25,   0.,     0.,     0.25,  -0.25,   0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
            [ 0.25, -0.25,  -0.25,   0.,     0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,     0.25,   0.25,   0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.25,  -0.25,   0.,    -0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,     0.25,  -0.25,   0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.25,   0.25,   0.,     0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,    -0.25,  -0.25,   0.,     0.,     0.,     0.,     0.   ],
            [ 0.25, -0.25,   0.25,   0.,    -0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,    -0.25,   0.25,   0.,     0.,     0.,     0.,     0.   ],
        ]),
    ]

    num_nodes_per_elem = 20
    num_surfaces = 6
    _num_gaussian = 27   # 3×3×3

    # Gauss-Legendre 3×3×3 weights (product of 1D weights)
    _w1d = torch.tensor([5.0/9.0, 8.0/9.0, 5.0/9.0])
    _x1d = torch.tensor([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gaussian_weight_ref = torch.ones(27)  # filled below
    gaussian_coordinates = torch.zeros([27, 3])  # filled below

    # Pre-compute the 3D Gauss points and weights
    _idx = 0
    for _i in range(3):
        for _j in range(3):
            for _k in range(3):
                gaussian_weight_ref[_idx] = _w1d[_i] * _w1d[_j] * _w1d[_k]
                gaussian_coordinates[_idx, 0] = _x1d[_i]
                gaussian_coordinates[_idx, 1] = _x1d[_j]
                gaussian_coordinates[_idx, 2] = _x1d[_k]
                _idx += 1
    del _idx, _i, _j, _k, _w1d, _x1d   # cleanup temporary loop variables

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        """
        Find surface elements for this element type
        
        Args:
            surface_ind: Surface index (0-5)
            elems_ind: Element indices to find surfaces for
            
        Returns:
            Tensor with surface node indices
        """
        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]
        
        if index_now.shape[0] == 0:
            quad_elems = torch.empty([0, 8], dtype=torch.long, device=self._elems.device)
            return [initialize_surfaces(quad_elems)]

        # Return appropriate face nodes according to face definitions in comments
        if surface_ind == 0:
            # Bottom face: 0-3-2-1 (nodes 0,3,2,1,11,10,9,8)
                quad_elems = self._elems[index_now][:, [0, 3, 2, 1, 11, 10, 9, 8]]
        elif surface_ind == 1:
            # Top face: 4-5-6-7 (nodes 4,5,6,7,12,13,14,15)
                quad_elems = self._elems[index_now][:, [4, 5, 6, 7, 12, 13, 14, 15]]
        elif surface_ind == 2:
            # Front face: 0-1-5-4 (nodes 0,1,5,4,8,17,12,16)
                quad_elems = self._elems[index_now][:, [0, 1, 5, 4, 8, 17, 12, 16]]
        elif surface_ind == 3:
            # Right face: 1-2-6-5 (nodes 1,2,6,5,9,18,13,17)
                quad_elems = self._elems[index_now][:, [1, 2, 6, 5, 9, 18, 13, 17]]
        elif surface_ind == 4:
            # Back face: 2-3-7-6 (nodes 2,3,7,6,10,19,14,18)
                quad_elems = self._elems[index_now][:, [2, 3, 7, 6, 10, 19, 14, 18]]
        elif surface_ind == 5:
            # Left face: 0-4-7-3 (nodes 0,4,7,3,16,15,19,11)
                quad_elems = self._elems[index_now][:, [0, 4, 7, 3, 16, 15, 19, 11]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
        
        return [initialize_surfaces(quad_elems)]

    def get_2nd_order_point_index_surface(self, surface_ind: int) -> torch.Tensor:
        """
        Get the 2nd order point index for the specified surface.
        This is used to identify the mid-edge nodes for the surface elements.
        
        Args:
            surface_ind: Surface index (0-5)
            
        Returns:
            torch.Tensor: Mid-edge node indices and their neighboring corner nodes
                size: [point_index, 3]
                [0]: the index of the middle node of the element
                [1]: the index of the neighbor node of the middle node of the element
                [2]: the index of the other neighbor node of the middle node of the element
        """
        if surface_ind == 0:
            # Bottom face: 0-3-2-1 with mid-edges 11,10,9,8
            return torch.tensor([[11, 0, 3],  # mid-edge between 0-3
                                [10, 2, 3],  # mid-edge between 3-2
                                [9, 1, 2],   # mid-edge between 2-1
                                [8, 0, 1]], dtype=torch.long, device='cpu')  # mid-edge between 1-0
        elif surface_ind == 1:
            # Top face: 4-5-6-7 with mid-edges 12,13,14,15
            return torch.tensor([[12, 4, 5],  # mid-edge between 4-5
                                [13, 5, 6],  # mid-edge between 5-6
                                [14, 6, 7],  # mid-edge between 6-7
                                [15, 4, 7]], dtype=torch.long, device='cpu')  # mid-edge between 7-4
        elif surface_ind == 2:
            # Front face: 0-1-5-4 with mid-edges 8,17,12,16
            return torch.tensor([[8, 0, 1],   # mid-edge between 0-1
                                [17, 1, 5],  # mid-edge between 1-5
                                [12, 4, 5],  # mid-edge between 5-4
                                [16, 0, 4]], dtype=torch.long, device='cpu')  # mid-edge between 4-0
        elif surface_ind == 3:
            # Right face: 1-2-6-5 with mid-edges 9,18,13,17
            return torch.tensor([[9, 1, 2],   # mid-edge between 1-2
                                [18, 2, 6],  # mid-edge between 2-6
                                [13, 5, 6],  # mid-edge between 6-5
                                [17, 1, 5]], dtype=torch.long, device='cpu')  # mid-edge between 5-1
        elif surface_ind == 4:
            # Back face: 2-3-7-6 with mid-edges 10,19,14,18
            return torch.tensor([[10, 2, 3],  # mid-edge between 2-3
                                [19, 3, 7],  # mid-edge between 3-7
                                [14, 6, 7],  # mid-edge between 7-6
                                [18, 2, 6]], dtype=torch.long, device='cpu')  # mid-edge between 6-2
        elif surface_ind == 5:
            # Left face: 0-4-7-3 with mid-edges 16,15,19,11
            return torch.tensor([[16, 0, 4],  # mid-edge between 0-4
                                [15, 4, 7],  # mid-edge between 4-7
                                [19, 3, 7],  # mid-edge between 7-3
                                [11, 0, 3]], dtype=torch.long, device='cpu')  # mid-edge between 3-0
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
