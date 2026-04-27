from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .assembly import Assembly

import numpy as np
import torch

from ..interfaces import Serializable


class BaseObj(Serializable):

    def __init__(self) -> None:

        super().__init__()

        """
        Initialize the FEA_Obj_Base class.
        """
        self._RGC_requirements: list[int] = [0]
        """
        The number of required RGCs for this object.
        """

        self._RGC_index: int = None
        """
        The index of the extra RGC for this object.
        """

        self._index_start: int = None
        """
        The start index of the extra RGC for this object
        """

        self._assembly: Assembly = None
        """The assembly this object belongs to."""

    @property
    def serialized_attributes(self):
        """Get the list of attributes to be serialized."""
        serialized_attrs = super().serialized_attributes
        serialized_attrs = [attr for attr in serialized_attrs if attr != '_assembly']
        serialized_attrs += ['_RGC_requirements']

        return serialized_attrs

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

    def initialize(self, assembly: Assembly):
        self._assembly = assembly
        self._index_start = assembly.RGC_list_indexStart[self._RGC_index]
        
    def initialize_dynamic(self):
        pass
    
    def reinitialize(self, RGC: list[torch.Tensor]):
        pass

