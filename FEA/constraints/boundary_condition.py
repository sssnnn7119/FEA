import numpy as np
import torch
from .base import BaseConstraint

class Boundary_Condition(BaseConstraint):
    """
    Boundary condition base class
    """

    def __init__(self,
                 indexDOF: np.ndarray = np.array([], dtype=int),
                 dispValue: torch.Tensor | float = 0.0,
                 rotational: bool = False,
                 rotation_axes: list = None) -> None:
        """
        Initialize boundary condition
        
        Parameters:
        -----------
        indexDOF : np.ndarray
            Indices of the degrees of freedom to constrain
        dispValue : torch.Tensor | float
            Displacement values for the constrained DOFs
        rotational : bool
            Whether to include rotational degrees of freedom
        rotation_axes : list
            List of rotation axes to include [rx, ry, rz]
            Each value should be True (free) or False (fixed)
            Default is [False, False, False] (all rotations fixed)
        """
        super().__init__()
        self.indexDOF = indexDOF
        
        if type(dispValue) != torch.Tensor:
            dispValue = dispValue * torch.ones([indexDOF.size])
        self.dispValue = dispValue
        
        # Rotational degrees of freedom
        self.rotational = rotational
        
        # Default: all rotations are fixed
        if rotation_axes is None:
            self.rotation_axes = [False, False, False]  # [rx, ry, rz]
        else:
            # Ensure we have exactly 3 values
            if len(rotation_axes) != 3:
                raise ValueError("rotation_axes must have exactly 3 values for rx, ry, and rz")
            self.rotation_axes = rotation_axes
    
    def get_rotation_dofs(self):
        """
        Get the rotational degrees of freedom status
        
        Returns:
        --------
        dict: Dictionary containing rotation status for each axis
        """
        return {
            "rotational": self.rotational,
            "rx": self.rotation_axes[0],
            "ry": self.rotation_axes[1],
            "rz": self.rotation_axes[2]
        }
    
    def set_rotation_dofs(self, rx=None, ry=None, rz=None):
        """
        Set which rotational degrees of freedom are free
        
        Parameters:
        -----------
        rx, ry, rz : bool or None
            True means free rotation around the axis
            False means fixed rotation around the axis
            None means don't change the current setting
        """
        self.rotational = True
        
        if rx is not None:
            self.rotation_axes[0] = rx
        if ry is not None:
            self.rotation_axes[1] = ry
        if rz is not None:
            self.rotation_axes[2] = rz
            
        return self

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply the boundary condition to the displacement vector
        """
        for i in range(len(self._fea.RGC_list_indexStart) - 1):
            index_now = (self.indexDOF >= self._fea.RGC_list_indexStart[i]) & (
                self.indexDOF < self._fea.RGC_list_indexStart[i + 1])
            # Convert dispValue to match RGC data type
            disp_values = self.dispValue[index_now].to(dtype=RGC[i].dtype)
            RGC[i].view(-1)[
                self.indexDOF[index_now] -
                self._fea.RGC_list_indexStart[i]] = disp_values
                
        # Apply rotational constraints if necessary
        if self.rotational and hasattr(self, 'rotation_indices'):
            for i, axis_free in enumerate(self.rotation_axes):
                if not axis_free and hasattr(self, f'rotation_indices_{i}'):
                    axis_indices = getattr(self, f'rotation_indices_{i}')
                    for j in range(len(self._fea.RGC_list_indexStart) - 1):
                        valid_indices = (axis_indices >= self._fea.RGC_list_indexStart[j]) & (
                            axis_indices < self._fea.RGC_list_indexStart[j + 1])
                        if np.any(valid_indices):
                            local_indices = axis_indices[valid_indices] - self._fea.RGC_list_indexStart[j]
                            RGC[j].view(-1)[local_indices] = 0.0
        
        return RGC

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        for i in range(len(self._fea.RGC_list_indexStart) - 1):
            index_now = (self.indexDOF >= self._fea.RGC_list_indexStart[i]) & (
                self.indexDOF < self._fea.RGC_list_indexStart[i + 1])
            RGC_remain_index[i].reshape(
                -1)[self.indexDOF[index_now] -
                    self._fea.RGC_list_indexStart[i]] = False
                    
        # Apply rotational constraints if necessary  
        if self.rotational and hasattr(self, 'rotation_indices'):
            for i, axis_free in enumerate(self.rotation_axes):
                if not axis_free and hasattr(self, f'rotation_indices_{i}'):
                    axis_indices = getattr(self, f'rotation_indices_{i}')
                    for j in range(len(self._fea.RGC_list_indexStart) - 1):
                        valid_indices = (axis_indices >= self._fea.RGC_list_indexStart[j]) & (
                            axis_indices < self._fea.RGC_list_indexStart[j + 1])
                        if np.any(valid_indices):
                            local_indices = axis_indices[valid_indices] - self._fea.RGC_list_indexStart[j]
                            RGC_remain_index[j].reshape(-1)[local_indices] = False
            
        return RGC_remain_index
        
    def set_rotation_indices(self, rx_indices=None, ry_indices=None, rz_indices=None):
        """
        Set the indices corresponding to rotational degrees of freedom
        
        Parameters:
        -----------
        rx_indices, ry_indices, rz_indices : np.ndarray
            Arrays containing indices for each rotational degree of freedom
        """
        if rx_indices is not None:
            self.rotation_indices_0 = rx_indices
        if ry_indices is not None:
            self.rotation_indices_1 = ry_indices
        if rz_indices is not None:
            self.rotation_indices_2 = rz_indices
            
        # Store all rotation indices together
        all_indices = []
        for i in range(3):
            if hasattr(self, f'rotation_indices_{i}'):
                all_indices.append(getattr(self, f'rotation_indices_{i}'))
        
        if all_indices:
            self.rotation_indices = np.concatenate(all_indices) if len(all_indices) > 1 else all_indices[0]
        
        return self
