import numpy as np
import torch
from .C3base import Element_3D


class C3D15(Element_3D):
    """
    # Local coordinates:
        origin: bottom triangle center
        g, h: coordinates in triangle base
        r: coordinate along prism height

    # Node numbering:
        - Bottom face (r=-1): 0, 1, 2 (vertices), 6, 7, 8 (mid-edge)
        - Top face (r=1): 3, 4, 5 (vertices), 9, 10, 11 (mid-edge)
        - Middle nodes (r=0): 12, 13, 14 (on vertical edges)

    # Face description:
        face0: 0(8)2(7)1(6) (Triangle)
        face1: 3(9)4(10)5(11) (Triangle)
        face2: 0(6)1(13)4(9)3(12) (Rectangle)
        face3: 1(7)2(14)5(10)4(13) (Rectangle)
        face4: 2(8)0(12)3(11)5(14) (Rectangle)

    # Shape functions:
        Quadratic interpolation in all directions
        Combines triangular base shape functions with prismatic extrusion
    """

    def __init__(self,
                 elems: torch.Tensor = None,
                 elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)
        self.order = 2
        # Shape function coefficients and derivatives
        # Format: [shape_function, derivatives]
        # These matrices represent the shape functions and their derivatives
        # for the 15-node prismatic element

    def initialize(self, fea):

        if self.order == 1:
            # Shape function coefficients in format aligned with your other elements
            self.shape_function = [
                torch.tensor([[0.5, -0.5, -0.5, -0.5, 0.0, 0.5, 0.5],
                              [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.5],
                              [0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0],
                              [0.5, -0.5, -0.5, 0.5, 0.0, -0.5, -0.5],
                              [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5],
                              [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0]]),
                torch.tensor([[[-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                               [0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                               [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                              [[-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                               [-0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]],
                              [[-0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                               [0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
                               [0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]]])
            ]
            self.num_nodes_per_elem = 6
            self._num_gaussian = 2
            self.gaussian_weight = torch.tensor([
                1 / 2,
                1 / 2,
            ])

            # get the interpolation coordinates of the guass_points
            p0 = torch.tensor([[1 / 3, 1 / 3, 1 / np.sqrt(3)],
                               [1 / 3, 1 / 3, -1 / np.sqrt(3)]])
        elif self.order == 2:
            # Shape function matrix (coefficients for each node's shape function)
            self.shape_function = [
                # Shape functions for all 15 nodes
                torch.tensor([[
                    0, -1.0, -1.0, -0.5, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0, 0,
                    -1.0, -0.5, -0.5, -1.0, -2.0, 0, 0, 0
                ],
                              [
                                  0, -1.0, 0, 0, 0, 0, 0.5, 1.0, 0, 0, 0, 0, 0,
                                  0, 0.5, -1.0, 0, 0, 0, 0
                              ],
                              [
                                  0, 0, -1.0, 0, 0, 0.5, 0, 0, 1.0, 0, 0, 0,
                                  -1.0, 0.5, 0, 0, 0, 0, 0, 0
                              ],
                              [
                                  0, -1.0, -1.0, 0.5, 2.0, -1.5, -1.5, 1.0,
                                  1.0, 0.5, 0, 0, 1.0, -0.5, -0.5, 1.0, 2.0, 0,
                                  0, 0
                              ],
                              [
                                  0, -1.0, 0, 0, 0, 0, -0.5, 1.0, 0, 0, 0, 0,
                                  0, 0, 0.5, 1.0, 0, 0, 0, 0
                              ],
                              [
                                  0, 0, -1.0, 0, 0, -0.5, 0, 0, 1.0, 0, 0, 0,
                                  1.0, 0.5, 0, 0, 0, 0, 0, 0
                              ],
                              [
                                  0, 2.0, 0, 0, -2.0, 0, -2.0, -2.0, 0, 0, 0,
                                  0, 0, 0, 0, 2.0, 2.0, 0, 0, 0
                              ],
                              [
                                  0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, -2.0, 0, 0, 0
                              ],
                              [
                                  0, 0, 2.0, 0, -2.0, -2.0, 0, 0, -2.0, 0, 0,
                                  0, 2.0, 0, 0, 0, 2.0, 0, 0, 0
                              ],
                              [
                                  0, 2.0, 0, 0, -2.0, 0, 2.0, -2.0, 0, 0, 0, 0,
                                  0, 0, 0, -2.0, -2.0, 0, 0, 0
                              ],
                              [
                                  0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 2.0, 0, 0, 0
                              ],
                              [
                                  0, 0, 2.0, 0, -2.0, 2.0, 0, 0, -2.0, 0, 0, 0,
                                  -2.0, 0, 0, 0, -2.0, 0, 0, 0
                              ],
                              [
                                  1.0, -1.0, -1.0, 0, 0, 0, 0, 0, 0, -1.0, 0,
                                  0, 0, 1.0, 1.0, 0, 0, 0, 0, 0
                              ],
                              [
                                  0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  -1.0, 0, 0, 0, 0, 0
                              ],
                              [
                                  0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  -1.0, 0, 0, 0, 0, 0, 0
                              ]]),
            ]

            self.shape_function.append(
                torch.stack([
                    self._shape_function_derivative(self.shape_function[0], 0),
                    self._shape_function_derivative(self.shape_function[0], 1),
                    self._shape_function_derivative(self.shape_function[0], 2),
                ],
                            dim=0))

            # Gauss weights

            gaussian_weight_triangle = torch.tensor([1 / 6, 1 / 6, 1 / 6])
            gaussian_points_triangle = torch.tensor([[1 / 6, 1 / 6],
                                                     [2 / 3, 1 / 6],
                                                     [1 / 6, 2 / 3]])

            gaussian_weight_height = torch.tensor([5 / 9, 8 / 9, 5 / 9])
            gaussian_points_height = torch.tensor(
                [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])

            # Combine weights and points for 3D integration
            self.gaussian_weight = torch.einsum(
                'i,j->ij', gaussian_weight_triangle,
                gaussian_weight_height).flatten()
            p0 = torch.cat([
                gaussian_points_triangle,
                torch.zeros([gaussian_points_triangle.shape[0], 1])
            ],
                           dim=1)
            p0 = p0.reshape([-1, 1, 3
                             ]).repeat([1, gaussian_points_height.shape[0], 1])
            p0[:, :, 2] = gaussian_points_height.reshape([1, -1])

            # Gauss integration points setup
            self.num_nodes_per_elem = 15
            self._num_gaussian = 9

        # Load the Gaussian points for integration
        self._pre_load_gaussian(p0.reshape([-1, 3]), nodes=fea.nodes)
        super().initialize(fea)

    def find_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]
        
        if index_now.shape[0] == 0:
            return torch.empty([0, 3], dtype=torch.long, device=self._elems.device)
        if self.order == 2:
            if surface_ind == 0:
                return torch.cat([self._elems[index_now][:, [0,8,6]],
                                self._elems[index_now][:, [8,2,7]],
                                self._elems[index_now][:, [6,7,1]],
                                self._elems[index_now][:, [6,8,7]],], dim=0)
                
            elif surface_ind == 1:
                return torch.cat([self._elems[index_now][:, [3,9,11]],
                                self._elems[index_now][:, [9,4,10]],
                                self._elems[index_now][:, [10,5,11]],
                                self._elems[index_now][:, [9,10,11]],], dim=0)
            elif surface_ind == 2:
                raise NotImplementedError(
                    "Surface 2 (Rectangle_2nd) is not implemented for C3D15")
            elif surface_ind == 3:
                raise NotImplementedError(
                    "Surface 3 (Rectangle_2nd) is not implemented for C3D15")
            elif surface_ind == 4:
                raise NotImplementedError(
                    "Surface 4 (Rectangle_2nd) is not implemented for C3D15")
            else:
                raise ValueError(f"Invalid surface index: {surface_ind}")
        elif self.order == 1:
            if surface_ind == 0:
                return self._elems[index_now][:, [0, 2, 1]]
            elif surface_ind == 1:
                return self._elems[index_now][:, [3, 4, 5]]
            elif surface_ind == 2:
                return torch.cat([self._elems[index_now][:, [0, 1, 4]], 
                                self._elems[index_now][:, [0, 4, 3]]], dim=1)
            elif surface_ind == 3:
                return torch.cat([self._elems[index_now][:, [1, 2, 5]],
                                self._elems[index_now][:, [1, 5, 4]]], dim=1)
            elif surface_ind == 4:
                return torch.cat([self._elems[index_now][:, [2, 0, 3]],
                                self._elems[index_now][:, [2, 3, 5]]], dim=1)
            else:
                raise ValueError(f"Invalid surface index: {surface_ind}")
            
            
            
    def refine_RGC(self, RGC: list[torch.Tensor], nodes: torch.Tensor) -> list[torch.Tensor]:
        
        mid_nodes_index = self.get_2nd_order_point_index()
        RGC[0][mid_nodes_index[:, 0]] = (RGC[0][mid_nodes_index[:, 1]] + RGC[0][mid_nodes_index[:, 2]]) / 2 + (nodes[mid_nodes_index[:, 1]] + nodes[mid_nodes_index[:, 2]] - 2 * nodes[mid_nodes_index[:, 0]]) / 2
        
        return RGC
    
    def set_order(self, order: int) -> None:
        self.order = order
        
    def get_2nd_order_point_index(self):
        mid_index =       self._elems[:, [6, 7, 8, 9, 10, 11, 12, 13, 14]].flatten()
        neighbor1_index = self._elems[:, [0, 1, 0, 3, 4,  3,  0,  1,  2]].flatten()
        neighbor2_index = self._elems[:, [1, 2, 2, 4, 5,  5,  3,  4,  5]].flatten()
        
        arg_index = torch.argsort(mid_index)
        
        
        result = torch.stack([mid_index, neighbor1_index, neighbor2_index], dim=1)
        result = result[arg_index]
        index_remain = torch.zeros([result.shape[0]], dtype=torch.bool, device='cpu')
        index_remain[0] = True
        index_remain[1:][result[1:, 0] > result[:-1, 0]] = True
        result = result[index_remain]

        return result
    
    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[0][self._elems[:, :6].unique()] = True
        if self.order > 1:
            RGC_remain_index[0][self._elems[:, 6:].unique()] = True
        return RGC_remain_index