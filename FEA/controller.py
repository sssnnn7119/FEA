import os
from re import T
import sys
import time
import ctypes
from unittest.mock import Base
import numpy as np
import torch
from .assemble import Assembly
from .solver import BaseSolver


class FEAController():

    def __init__(self, maximum_iteration: int = 10000) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """
        
        self.assembly: Assembly = None
        """The assembly containing instances, elements, and reference points."""

        self.solver: BaseSolver = None
        """The solver used for finite element analysis."""

    def initialize(self, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            None
        """
        self.assembly.initialize(*args, **kwargs)
        self.solver.initialize(assembly=self.assembly, *args, **kwargs)

    def solve(self, RGC0: torch.Tensor = None, tol_error: float = 1e-7, if_initialize: bool = True) -> bool:
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            bool: True if the solution converged, False otherwise.
        """
        if if_initialize:
            self.initialize(RGC0=RGC0)
        result = self.solver.solve(RGC0=RGC0, tol_error=tol_error)
        return result
