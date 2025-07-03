import numpy as np
import torch
from .base import BaseLoad
from ..elements import BaseSurface
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


        self.surface_element: list[BaseSurface]
        """
            the surface element
        """

        self.pressure = pressure
        """
        The pressure value applied to the surface element.
        """

    def initialize(self, fea):
        super().initialize(fea)
        
        self.surface_element = fea.get_surface_elements(self.surface_name)

        for surf_ind in range(len(self.surface_element)):
            if not isinstance(self.surface_element[surf_ind], BaseSurface):
                raise TypeError(f"Surface element {surf_ind} is not a valid BaseSurface instance.")
            self.surface_element[surf_ind].initialize(fea)
        
    def get_stiffness(self,
                RGC: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        U = RGC[0].reshape([-1, 3])

        F0_indices, F0_values, matrix_indices, values = self._get_K0_F0(U)

        Rf_values = self.pressure * F0_values

        return F0_indices, Rf_values, matrix_indices, values * self.pressure

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        
        node_pos = self._fea.nodes + RGC[0].reshape_as(self._fea.nodes)

        V = torch.zeros(1, dtype=self._fea.nodes.dtype, device=self._fea.nodes.device)
        for surf_ind in range(len(self.surface_element)):
            surf_elem = self.surface_element[surf_ind]
            if surf_elem._elems.shape[0] == 0:
                continue
            # evaluate the deformed position and its derivatives at the Gaussian points
            r_gaussian = torch.zeros([surf_elem._num_gaussian, surf_elem._elems.shape[0], 3],
                dtype=self._fea.nodes.dtype, device=self._fea.nodes.device)
            rdu_gaussian = torch.zeros([surf_elem._num_gaussian, surf_elem._elems.shape[0], 3, 2],
                dtype=self._fea.nodes.dtype, device=self._fea.nodes.device)
            for a in range(surf_elem.num_nodes_per_elem):
                r_gaussian += torch.einsum('g, ei->gei', surf_elem.shape_function_gaussian[0][:, a],
                                           node_pos[surf_elem._elems[:, a]])
                rdu_gaussian += torch.einsum('gm, ei->geim', surf_elem.shape_function_gaussian[1][:, :, a],
                                           node_pos[surf_elem._elems[:, a]])
            
            # evaluate the volume of the closed shell
            V_now = surf_elem.gaussian_weight.view([-1, 1]) * torch.cat([r_gaussian.unsqueeze(-1), rdu_gaussian], dim=-1).det() 
            V += V_now.sum()

        return -self.pressure * V

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        for surf_ind in range(len(self.surface_element)):

            RGC_remain_index[0][self.surface_element[surf_ind]._elems.flatten().unique().cpu()] = True
        return RGC_remain_index
