from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .Main import FEA_Main

import numpy as np
import torch


class BaseObj():

    def __init__(self) -> None:
        """
        Initialize the FEA_Obj_Base class.
        """
        self._RGC_requirements: int = 0
        """
        The number of required RGCs for this object.
        """

        self._RGC_index: int = None
        """
        The index of the extra RGC for this object.
        """

        self._fea: FEA_Main = None
        """
        The FEA_Main object that this object belongs to.
        """

    def set_RGC_index(self, index: int) -> None:
        """
        Set the index of the extra RGC for this object.
        """
        self._RGC_index = index

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        return RGC_remain_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return RGC

    def initialize(self, fea: FEA_Main):
        self._fea = fea
    
    def reinitialize(self, RGC: list[torch.Tensor]):
        pass