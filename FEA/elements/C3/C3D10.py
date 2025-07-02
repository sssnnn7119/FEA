import numpy as np
import torch
from .C3base import Element_3D

class C3D10(Element_3D):
    """
        Local coordinates:
            origin: 0-th nodal
            \ksi_0: 0-1 vector
            \ksi_1: 0-2 vector
            \ksi_2: 0-3 vector

        face nodal always point at the void
            face0: 0(6)2(5)1(4)
            face1: 0(4)1(8)3(7)
            face2: 1(5)2(9)3(8)
            face3: 0(7)3(9)2(6)

        2-nd element extra nodals:
            4(01) 5(12) 6(02) 7(03) 8(13) 9(23)

        shape_funtion:
            N_i = (2 \ksi_i - 1) * \ksi_i, i<=2 \n
            N_i = 4 \ksi_j \ksi_k, i>2 and jk is the neighbor nodals fo i-th nodal
    """

    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)
        self.order = 2
        
    def initialize(self, fea):
        
        if self.order == 1:
            
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

        elif self.order == 2:
            self.shape_function = [
                torch.tensor([[1., -3., -3., -3., 4., 4., 4., 2., 2., 2.],
                            [0., -1., 0., 0., 0., 0., 0., 2., 0., 0.],
                            [0., 0., -1., 0., 0., 0., 0., 0., 2., 0.],
                            [0., 0., 0., -1., 0., 0., 0., 0., 0., 2.],
                            [0., 4., 0., 0., -4., 0., -4., -4., 0., 0.],
                            [0., 0., 0., 0., 4., 0., 0., 0., 0., 0.],
                            [0., 0., 4., 0., -4., -4., 0., 0., -4., 0.],
                            [0., 0., 0., 4., 0., -4., -4., 0., 0., -4.],
                            [0., 0., 0., 0., 0., 0., 4., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 4., 0., 0., 0., 0.]]),
                torch.tensor([[[-3., 4., 4., 4., 0., 0., 0., 0., 0., 0.],
                            [-1., 4., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [4., -8., -4., -4., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 4., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., -4., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., -4., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 4., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                            [[-3., 4., 4., 4., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [-1., 0., 4., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., -4., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 4., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [4., -4., -8., -4., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., -4., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 4., 0., 0., 0., 0., 0., 0.]],
                            [[-3., 4., 4., 4., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [-1., 0., 0., 4., 0., 0., 0., 0., 0., 0.],
                            [0., -4., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., -4., 0., 0., 0., 0., 0., 0., 0.],
                            [4., -4., -4., -8., 0., 0., 0., 0., 0., 0.],
                            [0., 4., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 4., 0., 0., 0., 0., 0., 0., 0.]]])
            ]

            self.gaussian_weight = torch.tensor([1 / 24, 1 / 24, 1 / 24, 1 / 24])

            self.num_nodes_per_elem = 10
            self._num_gaussian = 4
            alpha = 0.58541020
            beta = 0.13819660

            p0 = torch.tensor([[beta, beta, beta], [alpha, beta, beta],
                            [beta, alpha, beta], [beta, beta, alpha]])
            
            
        self._pre_load_gaussian(p0, nodes=fea.nodes)
        super().initialize(fea)

    def refine_RGC(self, RGC: list[torch.Tensor], nodes: torch.Tensor) -> list[torch.Tensor]:
        
        mid_nodes_index = self.get_2nd_order_point_index()
        RGC[0][mid_nodes_index[:, 0]] = (RGC[0][mid_nodes_index[:, 1]] + RGC[0][mid_nodes_index[:, 2]]) / 2 + (nodes[mid_nodes_index[:, 1]] + nodes[mid_nodes_index[:, 2]] - 2 * nodes[mid_nodes_index[:, 0]]) / 2
        
        return RGC
    
    def set_order(self, order: int) -> None:
        self.order = order
    
    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[0][self._elems[:, :4].unique()] = True
        if self.order > 1:
            RGC_remain_index[0][self._elems[:, 4:].unique()] = True
        return RGC_remain_index

    def get_2nd_order_point_index(self):
        mid_index = torch.cat([self._elems[:, 4], self._elems[:, 5],
                      self._elems[:, 6], self._elems[:, 7],
                      self._elems[:, 8], self._elems[:, 9]])
        neighbor1_index = torch.cat([self._elems[:, 0], self._elems[:, 1], self._elems[:, 0], 
                           self._elems[:, 0], self._elems[:, 1], self._elems[:, 2]])
        neighbor2_index = torch.cat([self._elems[:, 1], self._elems[:, 2], self._elems[:, 2],
                            self._elems[:, 3], self._elems[:, 3], self._elems[:, 3]])
        
        arg_index = torch.argsort(mid_index)
        
        
        result = torch.stack([mid_index, neighbor1_index, neighbor2_index], dim=1)
        result = result[arg_index]
        index_remain = torch.zeros([result.shape[0]], dtype=torch.bool, device='cpu')
        index_remain[0] = True
        index_remain[1:][result[1:, 0] > result[:-1, 0]] = True
        result = result[index_remain]

        return result
      
    def find_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]
        
        if index_now.shape[0] == 0:
            return torch.empty([0, 3], dtype=torch.long, device=self._elems.device)
        
        if self.order == 2:
            if surface_ind == 0:
                return torch.cat([self._elems[index_now][:, [0,6,4]],
                                self._elems[index_now][:, [1,4,5]],
                                self._elems[index_now][:, [2,5,6]],
                                self._elems[index_now][:, [4,6,5]],], dim=0)
            elif surface_ind == 1:
                return torch.cat([self._elems[index_now][:, [0,4,7]],
                                self._elems[index_now][:, [1,8,4]],
                                self._elems[index_now][:, [3,7,8]],
                                self._elems[index_now][:, [4,8,7]],], dim=0)
            elif surface_ind == 2:
                return torch.cat([self._elems[index_now][:, [1,5,8]],
                                self._elems[index_now][:, [2,9,5]],
                                self._elems[index_now][:, [3,8,9]],
                                self._elems[index_now][:, [5,9,8]],], dim=0)
            elif surface_ind == 3:
                return torch.cat([self._elems[index_now][:, [0,7,6]],
                                self._elems[index_now][:, [2,6,9]],
                                self._elems[index_now][:, [3,9,7]],
                                self._elems[index_now][:, [7,9,6]],], dim=0)
            else:
                raise ValueError(f"Invalid surface index: {surface_ind}")

        elif self.order == 1:
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