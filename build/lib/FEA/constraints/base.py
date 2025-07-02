import numpy as np
import torch
from ..obj_base import BaseObj

class BaseConstraint(BaseObj):
    """
    Constraints base class
    """

    def __init__(self) -> None:
        """
        Initialize the Constraints_Base class.
        """
        super().__init__()

    def initialize(self, fea):
        super().initialize(fea)

    def modify_RGC_linear(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return self.modify_RGC(RGC)

    def modify_R_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
                   K_indices: torch.Tensor, K_values: torch.Tensor):

        R = torch.sparse_coo_tensor(indices=[[]],
                                    values=[],
                                    size=[self._fea.RGC_list_indexStart[-1]])
        return R, torch.zeros([2, 0], dtype=torch.int64), torch.zeros([0])

    def modify_R(self, RGC: list[torch.Tensor],
                 R0: torch.Tensor) -> torch.Tensor:
        R = torch.sparse_coo_tensor(indices=[[]],
                                    values=[],
                                    size=[self._fea.RGC_list_indexStart[-1]])
        return R

    def modify_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
                 K0: torch.Tensor):

        return torch.zeros([2, 0], dtype=torch.int64), torch.zeros([0])
