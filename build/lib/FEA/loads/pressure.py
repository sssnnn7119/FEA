import numpy as np
import torch
from .base import BaseLoad

class Pressure(BaseLoad):

    def __init__(self, surface_set: str, pressure: float) -> None:
        """
        initialize the pressure load on the surface element
        
        Args:
            surface_element (list[tuple[int, np.ndarray]]): the element index and the surface element index
            pressure (float): the pressure value
        """
        super().__init__()
        self.surface_name = surface_set
        self.surface_element: torch.Tensor
        """
            the surface element
        """
        self.pressure = pressure

    def initialize(self, fea):
        super().initialize(fea)
        
        surf = fea.get_surface_triangles(self.surface_name)
        self.surface_element = torch.cat(surf, dim=0)

        # for indices of force
        self._indices_force = self.surface_element.transpose(
            0, 1).unsqueeze(-1).repeat([1, 1, 3]).reshape([-1, 3])
        self._indices_force = self._indices_force * 3
        self._indices_force[:, 1] += 1
        self._indices_force[:, 2] += 2
        self._indices_force = self._indices_force.flatten().to(self._fea.nodes.device).to(torch.int64)

        # for indices of stiffness matrix
        index0 = torch.zeros([4, 0], dtype=torch.int, device='cpu')
        epsilon_indices = [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1],
                           [2, 1, 2, 0, 1, 0]]
        indice = self.surface_element.transpose(0, 1)
        for i in range(6):
            par_indice = [
                epsilon_indices[0][i] *
                torch.ones_like(indice[0], dtype=torch.int),
                epsilon_indices[1][i] *
                torch.ones_like(indice[0], dtype=torch.int),
                epsilon_indices[2][i] *
                torch.ones_like(indice[0], dtype=torch.int)
            ]
            index0 = torch.cat([
                index0,
                torch.stack(
                    [indice[0], par_indice[0], indice[1], par_indice[1]],
                    dim=0)
            ],
                               dim=1)
            index0 = torch.cat([
                index0,
                torch.stack(
                    [indice[1], par_indice[1], indice[0], par_indice[0]],
                    dim=0)
            ],
                               dim=1)
            index0 = torch.cat([
                index0,
                torch.stack(
                    [indice[1], par_indice[1], indice[2], par_indice[2]],
                    dim=0)
            ],
                               dim=1)
            index0 = torch.cat([
                index0,
                torch.stack(
                    [indice[2], par_indice[2], indice[1], par_indice[1]],
                    dim=0)
            ],
                               dim=1)
            index0 = torch.cat([
                index0,
                torch.stack(
                    [indice[2], par_indice[2], indice[0], par_indice[0]],
                    dim=0)
            ],
                               dim=1)
            index0 = torch.cat([
                index0,
                torch.stack(
                    [indice[0], par_indice[0], indice[2], par_indice[2]],
                    dim=0)
            ],
                               dim=1)
        index0[0] = index0[0] * 3 + index0[1]
        index0[2] = index0[2] * 3 + index0[3]
        index0 = index0[[0, 2]].to(torch.int64)

        # some trick to get the unique index and accelerate the calculation
        scaler = index0[0].max() + 1

        index1 = index0[0] * scaler + index0[1]
        index_sorted_matrix = index1.argsort()
        index2 = index1[index_sorted_matrix]
        index_unique, self._index_matrix_coalesce = torch.unique_consecutive(
            index2, return_inverse=True)
        self._indices_matrix = torch.zeros([2, index_unique.shape[0]],
                                          dtype=torch.int64)
        self._indices_matrix[
            1] = index_unique % scaler
        self._indices_matrix[
            0] = index_unique // scaler
        
        self._indices_matrix = self._indices_matrix.to(self._fea.nodes.device)

        inverse_index = torch.zeros_like(index_sorted_matrix,
                                         device='cpu',
                                         dtype=torch.int64)
        inverse_index[index_sorted_matrix] = torch.arange(
            0, index_sorted_matrix.max() + 1, device='cpu', dtype=torch.int64)
        self._index_matrix_coalesce = self._index_matrix_coalesce[inverse_index].to(self._fea.nodes.device)

    def get_stiffness(self,
                RGC: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        U = RGC[0].reshape([-1, 3])

        F0_indices, F0_values, matrix_indices, values = self._get_K0_F0(U)

        Rf_values = self.pressure * F0_values

        return F0_indices, Rf_values, matrix_indices, values * self.pressure

    def _get_K0_F0(self, U):
        surf_nodes = torch.stack(
            [(self._fea.nodes + U)[self.surface_element[:, i]]
             for i in [0, 1, 2]],
            dim=0)

        value = (-1 / 6) * torch.cat([
            torch.cross(surf_nodes[1], surf_nodes[2], dim=1),
            torch.cross(surf_nodes[2], surf_nodes[0], dim=1),
            torch.cross(surf_nodes[0], surf_nodes[1], dim=1)
        ],
                                     dim=0)

        epsilon_values = [1, -1, -1, 1, 1, -1]
        epsilon_indices = [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1],
                           [2, 1, 2, 0, 1, 0]]
        values = torch.zeros([0])
        for i in range(6):
            values = torch.cat([
                values,
                epsilon_values[i] * surf_nodes[2][:, epsilon_indices[2][i]]
            ],
                               dim=0)
            values = torch.cat([
                values,
                epsilon_values[i] * surf_nodes[2][:, epsilon_indices[2][i]]
            ],
                               dim=0)
            values = torch.cat([
                values,
                epsilon_values[i] * surf_nodes[0][:, epsilon_indices[0][i]]
            ],
                               dim=0)
            values = torch.cat([
                values,
                epsilon_values[i] * surf_nodes[0][:, epsilon_indices[0][i]]
            ],
                               dim=0)
            values = torch.cat([
                values,
                epsilon_values[i] * surf_nodes[1][:, epsilon_indices[1][i]]
            ],
                               dim=0)
            values = torch.cat([
                values,
                epsilon_values[i] * surf_nodes[1][:, epsilon_indices[1][i]]
            ],
                               dim=0)

        values = -torch.zeros(self._indices_matrix.shape[1]).scatter_add(
            0, self._index_matrix_coalesce, values) / 6

        return self._indices_force, value.flatten(), self._indices_matrix, values

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return -self.pressure * self.get_Volumn_Closed_Shell(
            self._fea.nodes + RGC[0].reshape_as(self._fea.nodes))

    def get_Volumn_Closed_Shell(self, nodes):
        node1 = nodes[self.surface_element[:, 0], :]
        node2 = nodes[self.surface_element[:, 1], :]
        node3 = nodes[self.surface_element[:, 2], :]
        Volumn = 1 / 6 * (torch.cross(node1, node2, dim=1) * node3).sum()
        return Volumn
  
    @staticmethod
    def get_F0(nodes, elems, max_RGC_index):
        
        surf_nodes = torch.stack(
            [nodes[elems[:, i]]
             for i in [0, 1, 2]],
            dim=0)

        value = (-1 / 6) * torch.cat([
            torch.cross(surf_nodes[1], surf_nodes[2], dim=1),
            torch.cross(surf_nodes[2], surf_nodes[0], dim=1),
            torch.cross(surf_nodes[0], surf_nodes[1], dim=1)
        ],
                                     dim=0)
        
        indices_force = elems.transpose(
            0, 1).unsqueeze(-1).repeat([1, 1, 3]).reshape([-1, 3])
        indices_force = indices_force * 3
        indices_force[:, 1] += 1
        indices_force[:, 2] += 2
        indices_force = indices_force.flatten()

        F0 = torch.zeros([max_RGC_index]).scatter_add(0, indices_force,
                                         value.flatten())
        
        return F0

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[0][self.surface_element.unique().cpu()] = True
        return RGC_remain_index
