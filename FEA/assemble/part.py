from __future__ import annotations
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Assembly

import numpy as np
import torch
from . import elements
from .obj_base import BaseObj
from .elements import BaseSurface, BaseElement, surfaces

class _Surfaces():
    """
    Class representing a set of surfaces in the finite element model.
    """

    def __init__(self):
        self._surface_dict: dict[str, list[tuple[np.ndarray, int]]] = {}
        self._surface_elements: dict[str, list[BaseSurface]] = {}
        self._initialized = False

    def initialize(self, part: Part):
        """
        Initialize the surface set before FEA.
        """
        self._surface_elements.clear()
        for name, surface_indices in self._surface_dict.items():
            element_now = part.extract_surfaces(name)
            self._surface_elements[name] = element_now

            # initialize the surface elements
            for se in element_now:
                se.initialize(part)
                self._initialized = True

    def get_elements(self, name: str) -> list[BaseSurface]:
        """
        Get the surface elements by their name.

        Args:
            name (str): The name of the surface.

        Returns:
            list[BaseSurface]: The list of surface elements.
        """
        return self._surface_elements.get(name, [])

    def __getitem__(self, key: str):
        """
        Get a surface by its name.

        Args:
            key (str): The name of the surface.

        Returns:
            list[tuple[np.ndarray, int]]: The surface data.
        """
        return self._surface_dict[key]

    def __setitem__(self, key: str, value: list[tuple[np.ndarray, int]]):
        """
        Set a surface by its name.

        Args:
            key (str): The name of the surface.
            value (list[tuple[np.ndarray, int]]): The surface data.
        """
        self._surface_dict[key] = value



    def __contains__(self, key: str):
        """
        Check if a surface exists by its name.

        Args:
            key (str): The name of the surface.

        Returns:
            bool: True if the surface exists, False otherwise.
        """
        return key in self._surface_dict

    def keys(self):
        """
        Get the keys of the surface set.

        Returns:
            list[str]: The list of surface names.
        """
        return list(self._surface_dict.keys())


class Part:
    def __init__(self, nodes: torch.Tensor) -> None:

        self.nodes: torch.Tensor = nodes
        """
        Nodes of the part.
        Shape: (num_nodes, 3)
        """
        self.elems: dict[str, BaseElement] = {}
        """
        Elements of the part.
        """

        self.surfaces = _Surfaces()

    def initialize(self, *args, **kwargs):
        for e in self.elems.values():
            e.initialize()
        self.surfaces.initialize(self)

    def add_element(self, element: BaseElement, name: str = None):
        """
        Add an element to the FEA model.

        Parameters:
            element (elements.Element_Base): The element to be added.

        Returns:
            str: The name of the element.
        """
        if name is None:
            number = len(self.elems)
            while ('element-%d' % number) in self.elems:
                number += 1
            name = 'element-%d' % number
        self.elems[name] = element
        return name

    def delete_element(self, name: str):
        """
        Delete an element from the FEA model.

        Parameters:
            name (str): The name of the element to be deleted.

        Returns:
            None
        """
        if name in self.elems:
            del self.elems[name]
        else:
            raise ValueError(f"Element '{name}' not found in the model.")
    
    def add_surface_set(self, name: str, elements: np.ndarray):
        """
        Add a surface set to the FEA model.
        
        Args:
            name (str): Name of the surface set.
            elements (np.ndarray): Surface elements information.
            
        Returns:
            str: Name of the added surface set.
        """
        self.surfaces[name] = elements
        return name
    
    def delete_surface_set(self, name: str):
        """
        Delete a surface set from the FEA model.
        
        Args:
            name (str): Name of the surface set to delete.
            
        Raises:
            KeyError: If the surface set doesn't exist.
        """
        if name in self.surfaces:
            del self.surfaces[name]
        else:
            raise KeyError(f"Surface set '{name}' not found in the model.")

    def refine_RGC(self, RGC: torch.Tensor) -> torch.Tensor:
        RGC_out = RGC
        for e in self.elems.values():
            RGC_out = e.refine_RGC(RGC_out, self.nodes)
        return RGC_out

    def merge_elements(self, element_name_list: list[str], element_name_new: str) -> None:
        """
        Merge multiple elements into a single new element.
        
        Args:
            element_name_list (list[str]): List of element names to merge.
            element_name_new (str): Name for the new merged element.
            
        Returns:
            str: Name of the merged element.
            
        Raises:
            ValueError: If elements are of different types or if any element name is not found.
        """
        if len(element_name_list) < 2:
            elems0 = self.elems[element_name_list[0]]
            self.add_element(elems0, name=element_name_new)
            self.delete_element(element_name_list[0])
            return
            
        # Check if all elements exist
        for name in element_name_list:
            if name not in self.elems:
                raise ValueError(f"Element '{name}' not found in the model")
            
        # Check if all elements are of the same type
        element_type = self.elems[element_name_list[0]].__class__.__name__
        for name in element_name_list[1:]:
            if self.elems[name].__class__.__name__ != element_type:
                raise ValueError(f"Cannot merge elements of different types: {type(self.elems[name])} and {element_type}")
        
        # Create a new element of the same type
        merged_elems = []
        merged_index = []
        for name in element_name_list:
            merged_elems.append(self.elems[name]._elems)
            merged_index.append(self.elems[name]._elems_index)
        merged_elems = torch.cat(merged_elems, dim=0)
        merged_index = torch.cat(merged_index, dim=0)
        merged_element = elements.initialize_element(element_type=element_type, elems_index=merged_index, elems=merged_elems, nodes=self.nodes)
        
        
        # Add merged element to the model
        self.add_element(merged_element, name=element_name_new)
        
        # Clean up the original elements if needed
        for name in element_name_list:
            self.delete_element(name)
            
        return

    def potential_Energy(self, RGC: torch.Tensor) -> torch.Tensor:
        p = torch.tensor(0.0)
        for e in self.elems.values():
            p = p + e.potential_Energy(RGC)
        return p

    def structural_Force(self, RGC: torch.Tensor, rotation_matrix: torch.Tensor) -> list[torch.Tensor]:
        K_values = []
        K_indices = []
        R_values = []
        R_indices = []
        for e in self.elems.values():
            Ra_indice, Ra_values, Ka_indice, Ka_value = e.structural_Force(
                RGC=RGC, rotation_matrix=rotation_matrix)
            K_values.append(Ka_value)
            K_indices.append(Ka_indice)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)

        K_indices = torch.cat(K_indices, dim=1)
        K_values = torch.cat(K_values, dim=0)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)
        return R_indices, R_values, K_indices, K_values
    
    def extract_surfaces(self, name: str) -> list[BaseSurface]:
        """        Get the triangles of a surface set by name.  
        
        Args:
            name (str): Name of the surface set.
            
        Returns:
            list[BaseSurface]: List of triangles in the surface set.
            
        Raises:
            ValueError: If the surface set is not found.
        """
        surface = []
        for surf_index in self.surfaces[name]:
            elem_ind = surf_index[0]
            surf_ind = surf_index[1]
            for e in self.elems.values():
                s_now = e.extract_surface(surf_ind, elem_ind)
                surface += s_now
        if len(surface) == 0:
            raise ValueError(f"Surface {surf_ind} not found in the model.")
        else:
            return surfaces.merge_surfaces(surface)

class Instance(BaseObj):
    def __init__(self, part: Part) -> None:
        """
        Create an instance of a part.

        Parameters:
            part (Part): The part to be instantiated.
        """
        self.part: Part = part

        self._RGC_requirements = self.part.nodes.shape

        self._translation: torch.Tensor = torch.zeros(3)
        self._rotation: torch.Tensor = torch.randn(3) * 0.0

    @property
    def rotation_matrix(self) -> torch.Tensor:
        theta = torch.norm(self._rotation)
        if theta == 0:
            return torch.eye(3)
        else:
            r = self._rotation / theta
            r = r.view(3, 1)
            R = torch.cos(theta) * torch.eye(3) + (1 - torch.cos(theta)) * (r @ r.t()) + torch.sin(theta) * torch.tensor([[0, -r[2, 0], r[1, 0]], [r[2, 0], 0, -r[0, 0]], [-r[1, 0], r[0, 0], 0]])
            return R

    @property
    def elems(self) -> dict[str, BaseElement]:
        return self.part.elems
    
    @property
    def nodes(self) -> torch.Tensor:
        return self._transform(self.part.nodes, self._rotation) + self._translation.unsqueeze(0)
    
    @property
    def surfaces(self) -> _Surfaces:
        return self.part.surfaces

    @staticmethod
    def _transform(vector0: torch.Tensor, rotation_vector: torch.Tensor = None):
        """
        Rotate a 3D vector by a rotation vector
        :param rotation_vector: rotation vector (3,)
        :param vector0: 3D vector (n, 3)
        :return: 3D vector (n, 3)
        """
        vector0 = vector0.view(-1, 3)
        if rotation_vector is None:
            return vector0
        
        theta = torch.norm(rotation_vector)
        if theta == 0:
            return vector0
        else:
            rotation_vector = rotation_vector / theta
            rotation_vector = rotation_vector.view(1, 3)
            vector1 = vector0 * torch.cos(theta) + torch.cross(
                rotation_vector, vector0, dim=1) * torch.sin(
                    theta) + rotation_vector * (rotation_vector * vector0).sum(
                        dim=1).unsqueeze(-1) * (1 - torch.cos(theta))
        return vector1

    def set_required_DoFs(self, RGC_remain_index):
        for e in self.part.elems.values():
            RGC_remain_index[self._RGC_index] = e.set_required_DoFs(RGC_remain_index[self._RGC_index])
        return RGC_remain_index
    
    def set_RGC_index(self, index):
        super().set_RGC_index(index)
    
    def initialize(self, assembly: Assembly):
        
        super().initialize(assembly=assembly)

    def refine_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        RGC_out = RGC
        RGC_out[self._RGC_index] = self.part.refine_RGC(RGC[self._RGC_index])
        return RGC_out
    
    def potential_Energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return self.part.potential_Energy(self._transform(rotation_vector=-self._rotation, vector0=RGC[self._RGC_index]))
    
    def structural_Force(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        R_indices, R_values, K_indices, K_values = self.part.structural_Force(RGC[self._RGC_index], self.rotation_matrix)
        return R_indices, R_values, K_indices, K_values
    
    def extract_surfaces(self, name: str) -> list[BaseSurface]:
        surfaces = self.part.extract_surfaces(name)
        return surfaces
    
    