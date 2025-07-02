from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..Main import FEA_Main

import numpy as np
import torch
from .. import elements
from ..obj_base import BaseObj





class BaseLoad(BaseObj):

    def __init__(self) -> None:
        super().__init__()
        self._indices_matrix: torch.Tensor = torch.zeros([2, 0],
                                                        dtype=torch.int)
        """
            the coo index of the stiffness matricx of structural stress
        """

        self._indices_force: torch.Tensor
        """
            the coo index of the tructural stress
        """

        self._index_matrix_coalesce: torch.Tensor = torch.zeros([0],
                                                            dtype=torch.int)
        """
            the start index of the stiffness matricx of structural stress
        """

    def initialize(self, fea: FEA_Main):
        super().initialize(fea)
    
    def get_stiffness(self,
                RGC: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        
        raise NotImplementedError("get_stiffness method not implemented")

    def get_potential_energy(self, RGC: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("get_potential_energy method not implemented")
    
    @staticmethod
    def get_F0():
        raise NotImplementedError("get_F0 method not implemented")
