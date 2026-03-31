from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pypardiso

if TYPE_CHECKING:
    from ... import Assembly

import numpy as np
import torch

from ..baseresult import BaseResult
import scipy.sparse as sp

class StaticResult(BaseResult):
    """
    The result of a static finite element analysis (FEA) simulation.
    """

    def __init__(self, GC: torch.Tensor, load_params: dict[str, torch.Tensor], jacobian: dict[str, torch.Tensor] = None, total_time: float = 0.0, time_items: dict[str, list[float]] = None):
        super().__init__()
        self.GC = GC.detach().clone()
        """ 
        Global displacements tensor
        """

        if jacobian is None:
            jacobian = {}
        else:
            jacobian = {k: v.detach().clone() for k, v in jacobian.items()}
        self.jacobian: dict[str, torch.Tensor] = jacobian
        """
        The Jacobian matrix (dGC/dLoadParams) of the static FEA result.
        """

        self.load_params: dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in load_params.items()}
        """
        Load parameters used in the simulation
        """

        self.K_solver: pypardiso.PyPardisoSolver = None
        """
        The Pardiso solver for the stiffness matrix
        """

        self.K_sp: sp.csr_matrix = None
        """
        The stiffness matrix in sparse CSR format
        """

        self.if_factorized: bool = False
        """
        Flag indicating whether the stiffness matrix has been factorized
        """

        self.total_time: float = total_time
        """Total time taken for the simulation"""

        self.time_items: dict[str, list[float]] = time_items if time_items is not None else {}
        """Dictionary to store time taken for different items in the simulation"""

    def __getstate__(self):
        """Exclude non-pickle-able objects (K_solver/K_sp) from serialized state."""
        state = self.__dict__.copy()
        state.pop('K_solver', None)
        state.pop('K_sp', None)
        return state

    def __setstate__(self, state):
        """Restore state and reset K_solver/K_sp to None after deserialization."""
        self.__dict__.update(state)
        self.K_solver = None
        self.K_sp = None

    def factorize_stiffness_matrix(self, assembly: 'Assembly'):
        """
        Factorize the stiffness matrix using the Pardiso solver.

        Args:
            K_idx (torch.Tensor): The indices of the stiffness matrix.
            K_val (torch.Tensor): The values of the stiffness matrix.
        """
        assembly.set_load_parameters(self.load_params)
        K_indices, K_values = assembly.assemble_Stiffness_Matrix(GC=self.GC)[1:]
        K_sp = sp.coo_matrix((K_values.detach().cpu().numpy(), (K_indices[0].cpu().numpy(), K_indices[1].cpu().numpy())))
        self.K_sp = K_sp.tocsr()
        self.K_solver = pypardiso.PyPardisoSolver()
        self.K_solver.factorize(self.K_sp)
        self.if_factorized = True

    def remove_stored_factorization(self):
        """
        Remove the stored factorization of the stiffness matrix to free up memory.
        """
        if self.K_solver is not None:
            self.K_solver.free_memory(everything=True)
            self.K_solver = None
            self.K_sp = None
            self.if_factorized = False
            import gc
            gc.collect()

    def save(self, path: str):
        """
        Save the static FEA result to a file.

        Args:
            path (str): The path to the file where the result will be saved.
        """
        # Implementation for saving static FEA results goes here
        load_params = {}
        if self.load_params is not None:
            for key, value in self.load_params.items():
                load_params[key] = value.cpu().numpy()

        np.savez_compressed(file=path, GC=self.GC.cpu().numpy(), load_params=load_params, total_time=self.total_time, time_items=self.time_items)
    
    @classmethod
    def load(cls, path: str) -> "StaticResult":
        """
        Load the static FEA result from a file.

        Args:
            path (str): The path to the file from which the result will be loaded.
        """
        # Implementation for loading static FEA results goes here
        data = np.load(path, allow_pickle=True)
        GC = torch.tensor(data['GC'])
        load_params_np = data['load_params'].item() if 'load_params' in data else None
        load_params = {}
        if load_params_np is not None:
            for key, value in load_params_np.items():
                load_params[key] = torch.tensor(value.tolist())
        total_time = float(data.get('total_time', 0.0))
        time_items = data.get('time_items', {}).item() if 'time_items' in data else {}
        return cls(GC=GC, load_params=load_params, total_time=total_time, time_items=time_items)