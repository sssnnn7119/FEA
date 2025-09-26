import numpy as np
import torch
from .base import BaseLoad

class Moment(BaseLoad):

    def __init__(self, rp_name: str, moment: list[float]) -> None:
        super().__init__()
        self.rp_name = rp_name
        self.rp_index: int = None
        self.moment = torch.tensor(moment)

    def initialize(self, assembly):
        super().initialize(assembly)
        self.rp_index = assembly.get_reference_point(self.rp_name)._RGC_index
        self._indices_force = torch.arange(assembly.RGC_list_indexStart[self.rp_index]+3, assembly.RGC_list_indexStart[self.rp_index]+6)

    def get_stiffness(self,
                RGC: list[torch.Tensor], if_onlyforce=False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:

        if if_onlyforce:
            return self._indices_force, self.moment

        return self._indices_force, self.moment, torch.zeros([2, 0], dtype=torch.int), torch.zeros([0])

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return (self.moment * RGC[self.rp_index][3:]).norm()

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self.rp_index][3:] = True
        return RGC_remain_index
