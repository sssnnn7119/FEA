import numpy as np
import torch
from .base import BaseConstraint

class Boundary_Condition(BaseConstraint):
    """
    Boundary condition base class
    """

    def __init__(self,
                 instance_name: str,
                 index_nodes: np.ndarray,
                 indexDoF: list[int] = [0, 1, 2],
                 ) -> None:
        """
        Initialize boundary condition
        
        """
        super().__init__()
        self.index_nodes = np.sort(list(index_nodes))
        self.instance_name = instance_name
        self.indexDoF = indexDoF
        """Record the instance name"""

        self._constraint_index: int

    def initialize(self, assembly):
        super().initialize(assembly)
        self._constraint_index = self._assembly.get_instance(self.instance_name)._RGC_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply the boundary condition to the displacement vector
        """
        for i in self.indexDoF:
            RGC[self._constraint_index][self.index_nodes, i] = 0.0

        return RGC

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        for i in self.indexDoF:
            RGC_remain_index[self._constraint_index][self.index_nodes, i] = False
            
        return RGC_remain_index
