from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Assembly, Part

import numpy as np
import torch
from . import materials
class BaseElement():
    _subclasses: dict[str, 'BaseElement'] = {}

    def __init_subclass__(cls):
        """Register subclasses in the class registry for factory method."""
        cls._subclasses[cls.__name__] = cls

    def __init__(self, elems_index: torch.Tensor, elems: torch.Tensor, part: Part) -> None:

        super().__init__()
        self.shape_function: list[torch.Tensor]
        """
            the shape of shape_function 
        """
        
        self.shape_function_gaussian: list[torch.Tensor]
        """
            the shape of shape_function at each guassian point
        """
        
        self.gaussian_weight: torch.Tensor
        """
        the weight of each guassian point
            [
                g, the num of guassian point
            ]
        """
        self._elems_index = elems_index
        """
            the index of the element
        """
        self._elems = elems
        """
            [elem, N]\n
            the element connectivity 
        """

        self._num_gaussian: int

        self.materials: materials.Materials_Base

        self._indices_matrix: torch.Tensor
        """
            the coo index of the stiffness matricx of structural stress
        """

        self._indices_force: torch.Tensor
        """
            the coo index of the tructural stress
        """

        self._index_matrix_coalesce: torch.Tensor
        """
            the start index of the stiffness matricx of structural stress
        """

        self.density: torch.Tensor
        """
            the density of the element
        """
        self.num_nodes_per_elem: int
        """
            the number of nodes per element
        """

        self.part = part

    def get_gaussian_points(self, nodes: torch.Tensor) -> torch.Tensor:
        """
            get the gaussian points of the element
        """
        raise NotImplementedError('The gaussian points of the element is not implemented yet')

    def initialize(self, *args, **kwargs):
        pass
    
    def potential_Energy(self, RGC: torch.Tensor):
        pass

    def structural_Force(self, RGC: torch.Tensor, *args, **kwargs):
        pass

    def set_materials(self, materials: materials.Materials_Base):
        """
            set the materials of the element
        """
        
        self.materials = materials

    def set_density(self, density: torch.Tensor |float):
        """
            set the density of the element
        """
        if type(density) == float:
            density = torch.tensor([density], dtype=torch.float32)

        self.density = density
        
    def extract_surface(self, surface_ind: int, elems_ind: np.ndarray):
        """
        Find the surface of the element

        Args:
            surface_ind (int): the index of the surface
            elems_ind (np.ndarray): the index of the element
        
        Returns:
            list[BaseSurface]: a list of surface elements
        """
        return []
    
    def set_order(self, order: int):
        """
        set the order of the element
        Args:
            order (int): the order of the element
        """
        raise NotImplementedError('The order of the element is not implemented yet')
    
    def refine_RGC(self, RGC: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
            refine the RGC of the element
        """
        return RGC