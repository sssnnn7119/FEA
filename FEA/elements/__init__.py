import numpy as np
import torch
from .base import BaseElement
from .C3 import C3D4, C3D6, C3D8R, C3D10, C3D15, C3D8, C3D20
from . import materials


# methods for the elements
from .generate_shell import generate_shell_from_surface, add_shell_elements_to_model
from .convert_elements import convert_to_second_order

def initialize_element(element_type: str,
                       elems_index: torch.Tensor, elems: torch.Tensor, *args,
                       **kwargs) -> BaseElement:
    """
    Initialize the element based on the element type.

    Args:
        element_type (str): The type of the element to initialize.
        elems (np.ndarray): The elements array containing element connectivity.
        nodes (torch.Tensor): The nodes array containing node coordinates.
        materials (torch.Tensor): The materials array containing material properties.

    Returns:
        Element_Base: An instance of the specified element type.
    """
    element_class = BaseElement._subclasses.get(element_type)
    
    if element_class is None:
        raise ValueError(f"Element type '{element_type}' is not recognized.")
    
    
    result = element_class(elems_index=elems_index, elems=elems)

    return result
