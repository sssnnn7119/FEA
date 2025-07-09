import os
import sys
import time
import ctypes
import numpy as np
import torch

from . import loads
from . import elements
from . import constraints
from .elements.C3 import surfaces
from .reference_points import ReferencePoint
from .solver import linear_solver as _linear_Solver
from .solver import lbfgs as _lbfgs

class FEA_Main():
    """
    Main class for Finite Element Analysis (FEA).
    This class handles the core functionality of finite element analysis, including
    model initialization, solving, and post-processing. It manages nodes, elements,
    materials, loads, and constraints to simulate structural behavior.
    Attributes:
        nodes (list): List containing node coordinates and reference points.
        elems (dict): Dictionary mapping element types to Element objects.
        RGC_list_indexStart (list): Start indices for redundant generalized coordinates.
        _RGC_nameMap (dict): Maps RGC indices to names.
        _RGC_size (dict): Maps RGC indices to size information.
        loads (dict): Dictionary of applied loads.
        constraints (dict): Dictionary of applied constraints.
        RGC (list): Redundant generalized coordinates.
        RGC_remain_index (list): Boolean masks for remaining indices.
        GC (torch.Tensor): Generalized coordinates.
        GC_list_indexStart (list): Start indices for generalized coordinates.
        RGC_remain_index_flatten (torch.Tensor): Flattened boolean mask for remaining indices.
        node_sets (dict): Dictionary of node sets imported from FEA_INP.
        element_sets (dict): Dictionary of element sets imported from FEA_INP.
        surface_sets (dict): Dictionary of surface sets imported from FEA_INP.
    Methods:
        initialize: Initialize the finite element model.
        solve: Solve the finite element analysis problem.
        assemble_force: Assemble the global force vector.
        add_reference_point: Add a reference point to the model.
        add_load: Add a load to the model.
        add_constraint: Add a constraint to the model.
        solve_linear_perturbation: Solve a linear perturbation problem.
        add_node_set: Add a node set to the model.
        add_element_set: Add an element set to the model.
        add_surface_set: Add a surface set to the model.
        get_node_set: Get a node set by name.
        get_element_set: Get an element set by name.
        get_surface_set: Get a surface set by name.
    """

    def __init__(self, nodes: torch.Tensor, maximum_iteration: int = 100) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """
        self.nodes: torch.Tensor = nodes
        """
        record the nodes of the finite element model.\n
        """

        self.RGC_list_indexStart: list[int]

        # initialize the reference points
        self.reference_points: dict[str, ReferencePoint] = {}

        # initialize the elements
        self.elems: dict[str, elements.BaseElement] = {}
        """
        record the elements of the finite element model.\n
        """

        # initialize the loads
        self.loads: dict[str, loads.BaseLoad] = {}

        # initialize the constraints
        self.constraints: dict[str, constraints.BaseConstraint] = {}

        # initialize sets collections
        self.node_sets: dict[str, np.ndarray] = {}
        self.element_sets: dict[str, np.ndarray] = {}
        self.surface_sets: dict[str, np.ndarray] = {}

        self._RGC_nameMap: dict[int, str]
        """
        record the name of the RGC\n
        {0: 'nodes_disp', 1: 'nodes_rot'}
        """

        self._RGC_size: dict[int, list[int]]
        """
        record the size of the RGC\n
        {0: nodes.shape, 1: nodes.shape}
        """

        self.RGC: list[torch.Tensor]
        """
        record the redundant generalized coordinates\n
        [0]: translation\n
        [1]: orientation\n
        [2]: allocated for other objects\n
        """

        self.RGC_remain_index: list[np.ndarray]
        """
        record the remaining index of the RGC\n
        """

        self.RGC_remain_index_flatten: torch.Tensor
        """
        record the remaining index of the RGC (flattened)\n
        """

        # initialize the GC (generalized coordinates)
        self.GC: torch.Tensor
        """
        record the generalized coordinates\n
        """

        self._GC_list_indexStart: list[int] = []
        """
        record the start index of the GC\n
        """

        self.maximum_iteration: int = maximum_iteration
        """
        the allowed maximum number of iterations for the solver.\n
        """

        self._iter_now: int = 0
        """
        The iteration of the FEA step
        """

    def initialize(self, RGC0: torch.Tensor = None):
        """
        Initialize the finite element model.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            None
        """

        # region initialize the RGC

        # initialize the RGC (redundant generalized coordinate)
        self.RGC: list[torch.Tensor] = [
            torch.zeros_like(self.nodes),
            torch.zeros_like(self.nodes)
        ]

        self.RGC_remain_index: list[np.ndarray] = [
            np.zeros(self.nodes.shape, dtype=bool),
            np.zeros(self.nodes.shape, dtype=bool)
        ]

        self._RGC_nameMap: dict[int, str] = {0: 'nodes_disp', 1: 'nodes_rot'}
        self._RGC_size: dict[int, list[int]] = {
            0: self.nodes.shape,
            1: self.nodes.shape
        }

        for rp in self.reference_points.keys():
            RGC_index = self._allocate_RGC(
                size=self.reference_points[rp]._RGC_requirements, name=rp)
            self.reference_points[rp].set_RGC_index(RGC_index)

        for e in self.elems.keys():
            RGC_index = self._allocate_RGC(
                size=self.elems[e]._RGC_requirements, name=e)
            self.elems[e].set_RGC_index(RGC_index)

        for f in self.loads.keys():
            RGC_index = self._allocate_RGC(
                size=self.loads[f]._RGC_requirements, name=f)
            self.loads[f].set_RGC_index(RGC_index)

        for c in self.constraints.keys():
            RGC_index = self._allocate_RGC(
                size=self.constraints[c]._RGC_requirements, name=c)
            self.constraints[c].set_RGC_index(RGC_index)

        self.RGC_list_indexStart = [0]
        for i in range(len(self.RGC)):
            self.RGC_list_indexStart.append(self.RGC_list_indexStart[i] +
                                            self.RGC[i].numel())

        if RGC0 is not None:
            for i in range(min(len(self.RGC), len(RGC0))):
                self.RGC[i] = RGC0[i].clone().detach()
        # endregion

        # region initialize the elements, loads, and constraints
        # initialize the elements
        for e in self.elems.values():
            e.initialize(self)

        # initialize the loads
        for l in self.loads.values():
            l.initialize(self)

        # initialize the constraints
        for c in self.constraints.values():
            c.initialize(self)

        # endregion

        # region modify the RGC_remain_index
        for e in self.elems.values():
            self.RGC_remain_index = e.set_required_DoFs(self.RGC_remain_index)

        for f in self.loads.values():
            self.RGC_remain_index = f.set_required_DoFs(self.RGC_remain_index)

        for c in self.constraints.values():
            self.RGC_remain_index = c.set_required_DoFs(self.RGC_remain_index)

        self.RGC_remain_index_flatten = np.concatenate([
            self.RGC_remain_index[i].reshape(-1)
            for i in range(len(self.RGC_remain_index))
        ]).tolist()
        self.RGC_remain_index_flatten = torch.tensor(
            self.RGC_remain_index_flatten, dtype=torch.bool)

        # GC core
        self.GC = self._RGC2GC(self.RGC)
        self._GC_list_indexStart = np.cumsum([
            self.RGC_remain_index[j].sum()
            for j in range(len(self.RGC_remain_index))
        ]).tolist()
        self._GC_list_indexStart.insert(0, 0)

        # endregion

    def solve(self, RGC0: torch.Tensor = None, tol_error: float = 1e-7) -> bool:
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            bool: True if the solution converged, False otherwise.
        """
        # initialize the RGC
        self.initialize(RGC0=RGC0)

        # initialize the iteration

        t0 = time.time()
        # start the iteration
        result = self._solve_iteration(RGC=self.RGC,
                                         tol_error=tol_error)
        self.RGC = self._GC2RGC(self.GC)

        self.refine_RGC()
        t1 = time.time()

        # print the information
        print('total_iter:%d, total_time:%.2f' % (self._iter_now, t1 - t0))
        R = self._assemble_Stiffness_Matrix(RGC = self.RGC)[0]
        print('max_error:%.4e' % (R.abs().max()))
        print('---' * 4, 'FEA Finished', '---' * 4, '\n')

        return result

    # region solve iteration

    def assemble_force(self, force, GC0: torch.Tensor = None) -> torch.Tensor:
        if force.dim() == 1:
            force = force.unsqueeze(0)
        R = force.clone()
        if GC0 is None:
            GC0 = self.GC
        RGC = self._GC2RGC(GC0)
        for c in self.constraints.values():
            for i in range(R.shape[0]):
                R_new = c.modify_R(RGC, force[i].flatten())
                R[i] += R_new

        R = R[:, self.RGC_remain_index_flatten]

        return R

    def _assemble_Stiffness_Matrix(self,
                                   RGC: list[torch.Tensor],):
        """
        Assemble the stiffness matrix.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.

        Returns:
            tuple: A tuple containing the right-hand side vector, the indices of the stiffness matrix, and the values of the stiffness matrix.
                -
        """
        #region evaluate the structural K and R
        R0, K_indices, K_values = self._assemble_generalized_Matrix(
            RGC)
        # endregion
        R, K_indices, K_values = self._assemble_reduced_Matrix(
            RGC, R0, K_indices, K_values)

        return R, K_indices, K_values

    def _assemble_generalized_Matrix(self,
                                     RGC: list[torch.Tensor],):

        #region evaluate the structural K and R
        t0 = time.time()
        K_values = []
        K_indices = []
        R_values = []
        R_indices = []

        for e in self.elems.values():
            Ra_indice, Ra_values, Ka_indice, Ka_value = e.structural_Force(
                RGC=RGC)
            K_values.append(Ka_value)
            K_indices.append(Ka_indice)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)
        t1 = time.time()
        for f in self.loads.values():
            Rf_indice, Rf_values, Kf_indice, Kf_value = f.get_stiffness(
                RGC=RGC)
            K_values.append(-Kf_value)
            K_indices.append(Kf_indice)
            R_values.append(-Rf_values)
            R_indices.append(Rf_indice)
        t2 = time.time()
        # endregion

        K_indices = torch.cat(K_indices, dim=1)
        K_values = torch.cat(K_values, dim=0)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)

        R0 = torch.zeros(self.RGC_list_indexStart[-1])
        # Convert R_indices to int64 explicitly for scatter operation
        R0.scatter_add_(0, R_indices.to(torch.int64), R_values)
        return R0, K_indices, K_values

    def _assemble_reduced_Matrix(self, RGC: list[torch.Tensor],
                                 R0: torch.Tensor, K_indices: torch.Tensor,
                                 K_values: torch.Tensor):
        t0 = time.time()
        R = R0.clone()
        #region consider the constraints
        for c in self.constraints.values():
            R_new, Kc_indices, Kc_values = c.modify_R_K(
                RGC, R0, K_indices, K_values)
            K_indices = torch.cat([K_indices, Kc_indices], dim=1)
            K_values = torch.cat([K_values, Kc_values])
            R += R_new
        t4 = time.time()
        #endregion

        # get the global stiffness matrix and force vector
        index_remain = self.RGC_remain_index_flatten[K_indices[0].cpu(
        )] & self.RGC_remain_index_flatten[K_indices[1].cpu()]
        K_values = K_values[index_remain]
        K_indices = K_indices[:, index_remain]
        t44 = time.time()

        K_indices[0] = K_indices[0].unique(return_inverse=True)[1]
        K_indices[1] = K_indices[1].unique(return_inverse=True)[1]

        t5 = time.time()

        R = R[self.RGC_remain_index_flatten]

        t6 = time.time()
        return R, K_indices, K_values

    def _total_Potential_Energy(self,
                                RGC: list[torch.Tensor]):
        """
        Calculate the total potential energy of the finite element model.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.

        Returns:
            float: The total potential energy.
        """

        # structural energy
        energy = 0
        for e in self.elems.values():
            energy = energy + e.potential_Energy(RGC=RGC)

        # force potential
        for f in self.loads.values():
            energy = energy - f.get_potential_energy(RGC=RGC)

        return energy

    def _line_search(self,
                     GC0: torch.Tensor,
                     dGC: torch.Tensor,
                     R: torch.Tensor,
                     energy0: float, *args, **kwargs):
        # line search
        alpha = 1
        beta = float('inf')
        c1 = 0.02
        c2 = 0.4
        dGC0 = dGC.clone()
        deltaE = (dGC * R).sum()

        if deltaE > 0:
            dGC = -dGC
            deltaE = -deltaE

        if torch.isnan(dGC).sum() > 0 or torch.isinf(dGC).sum() > 0:
            dGC = -R
            deltaE = (dGC * R).sum()

        # if abs(deltaE / energy0) < tol_error:
        #     return 1, GC0

        loopc2 = 0
        while True:
            GCnew = GC0 + alpha * dGC
            # GCnew.requires_grad_()
            energy_new = self._total_Potential_Energy(
                RGC=self._GC2RGC(GCnew))

            if torch.isnan(energy_new) or torch.isinf(
                    energy_new) or energy_new > energy0 + c1 * deltaE * alpha:
                alpha = 0.5 * alpha
                if alpha < 1e-12:
                    alpha = 0.0
                    GCnew = GC0.clone()
                    energy_new = energy0
                    break
            else:
                # Rnew = -torch.autograd.grad(energy_new, GCnew)[0]
                # if torch.dot(Rnew, dGC) > c2 * deltaE:
                #     beta = alpha
                #     alpha = 0.6 * (alpha + beta)
                # elif torch.dot(Rnew, dGC) < -c2 * deltaE:
                #     beta = alpha
                #     alpha = 0.4 * (alpha + beta)
                # else:
                break
            loopc2 += 1
            if loopc2 > 20:
                c2 = 1000000000000000

        # if abs(alpha) < 1e-3:
        #     # gradient direction line search
        #     alpha = 1
        #     dGC = R
        #     while True:
        #         GCnew = GC0 + alpha * dGC
        #         energy_new = self._total_Potential_Energy(
        #             RGC=self._GC2RGC(GCnew))
        #         if energy_new < energy0:
        #             # pressure *= 1.2
        #             # pressure = min(pressure0, pressure)
        #             break
        #         alpha *= 0.8
        #         if abs(alpha) < 1e-10:
        #             alpha = 0.0
        #             GCnew = GC0.clone()
        #             energy_new = energy0
        #             break

        # if abs(alpha) < 1e-3:
        #     alpha = 1
        #     GCnew = GC0 + alpha * dGC0
        return alpha, GCnew.detach(), energy_new.detach()

    def _solve_iteration(self,
                         RGC: list[torch.Tensor],
                         tol_error: float):

        GC = self._RGC2GC(RGC)
        RGC = self._GC2RGC(GC)

        # iteration now
        self._iter_now = 0

        # initialize the time
        t00 = time.time()

        # initialize the energy
        energy = [
            self._total_Potential_Energy(RGC=RGC)
        ]

        # check the initial energy, if nan, reinitialize the RGC
        if torch.isnan(energy[-1]):
            for i in range(len(self.RGC)):
                RGC[i] = torch.randn_like(self.RGC[i]) * 1e-10

            GC = self._RGC2GC(RGC)
            energy[-1] = self._total_Potential_Energy(
                RGC=RGC)
        dGC = torch.zeros_like(GC)

        # record the number of low alpha
        low_alpha = 0
        alpha = 0

        # begin the iteration
        while True:

            if self._iter_now > self.maximum_iteration:
                print('maximum iteration reached')
                return False

            # calculate the force vector and tangential stiffness matrix
            t1 = time.time()
            R, K_indices, K_values = self._assemble_Stiffness_Matrix(
                RGC=RGC)

            

            self._iter_now += 1

            # evaluate the newton direction
            t2 = time.time()
            dGC = self._solve_linear_equation(K_indices=K_indices,
                                              K_values=K_values,
                                              R=-R,
                                              iter_now=self._iter_now,
                                              alpha0=alpha,
                                              tol_error=tol_error,
                                              dGC0=dGC).flatten()




            # line search
            t3 = time.time()
            alpha, GCnew, energynew = self._line_search(
                GC, dGC, R, energy[-1])

            if alpha==0 and R.abs().max() > tol_error:
                return False

            # if convergence has difficulty, reduce the load percentage
            if alpha < 0.1:
                low_alpha += 1
            else:
                low_alpha -= 5
                if low_alpha < 0:
                    low_alpha = 0

            if low_alpha > 50:
                if R.abs().max() < 1e-3:
                    print('low alpha, but convergence achieved')
                    break
                return False

            # update the GC
            GC = GCnew

            # update the RGC
            RGC = self._GC2RGC(GC)

            # update the energy
            energynew = self._total_Potential_Energy(
                RGC=RGC)
            energy.append(energynew)

            t4 = time.time()

            # return the index to the first line
            if self._iter_now > 0:
                print('\033[1A', end='')
                print('\033[1A', end='')
                print('\033[K', end='')

            print(  "{:^8}".format("iter") + \
                    "{:^8}".format("alpha") + \
                    "{:^15}".format("total") + \
                    "{:^15}".format("energy") + \
                    "{:^15}".format("error") + \
                    "{:^15}".format("assemble") + \
                    "{:^15}".format("linearEQ") + \
                    "{:^15}".format("line search") + \
                    "{:^15}".format("step"))

            print(  "{:^8}".format(self._iter_now) + \
                    "{:^8.2f}".format(alpha) + \
                    "{:^15.2f}".format(t4 - t00) + \
                    "{:^15.4e}".format(energy[-1]) + \
                    "{:^15.4e}".format(R.abs().max()) + \
                    "{:^15.2f}".format(t2 - t1) + \
                    "{:^15.2f}".format(t3 - t2) + \
                    "{:^15.2f}".format(t4 - t3) + \
                    "{:^15.2f}".format(t4 - t1))
            
            if dGC.abs().max() < tol_error:
                break
        self.GC = GC
        return True

    __low_alpha_count = 0

    def _solve_mixed_method(self,
                         RGC: list[torch.Tensor],
                         tol_error: float):
        """
        Solve the finite element analysis problem using a mixed method.
        if the number of iterations is less than 10, use lbfgs method,
        otherwise use newton method.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
            tol_error (float): The tolerance error for convergence.

        Returns:
            bool: True if the solution converged, False otherwise.
        """

        GC = self._RGC2GC(RGC)

        # iteration now
        iter_now = 0
        max_iterations = 100
        tol_convergence = tol_error
        energy = [self._total_Potential_Energy(RGC=RGC)]
        low_alpha_count = 0
        stagnation_counter = 0
        last_error = float('inf')
        method = "lbfgs"  # Start with Newton method
        
        t00 = time.time()
        
        # Initialize dGC for tracking
        step = torch.zeros_like(GC)
        
        while iter_now < max_iterations:
            # Calculate current error
            R, K_indices, K_values = self._assemble_Stiffness_Matrix(
            RGC=self._GC2RGC(GC))
            current_error = R.abs().max().item()
            
            # Print iteration header (only at start or when method changes)
            if iter_now == 0 or method_changed:
                print(f"\nUsing {method.upper()} method")
                print("{:^8} {:^10} {:^15} {:^15} {:^15}".format(
                    "Iter", "Method", "Energy", "Max Error", "Time (s)"))
                method_changed = False
            
            # Print current iteration status
            print("{:^8d} {:^10} {:^15.4e} {:^15.4e} {:^15.2f}".format(
            iter_now, method, energy[-1], current_error, time.time() - t00))
            
            # Check convergence
            if current_error < tol_convergence:
                print(f"\nConverged after {iter_now} iterations!")
                break
            
            # Check for stagnation
            if abs(current_error - last_error) < tol_convergence * 0.01:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_error = current_error
            
            # Method switching logic
            method_changed = False
            if method == "newton" and (stagnation_counter > 3 or low_alpha_count > 2):
                method = "lbfgs"
                method_changed = True
                stagnation_counter = 0
                low_alpha_count = 0
            elif method == "lbfgs" and iter_now % 5 == 4:  # Switch back to Newton occasionally
                method = "newton"
                method_changed = True
            
            # Step with the selected method
            if method == "newton":
                # Newton step
                energynew, step, alpha = self._step_newton(GC, step, energy, iter_now, sub_iter_num=10)
                # Track small step sizes
                if alpha < 0.01:
                    low_alpha_count += 1
            else:
                # L-BFGS step
                energynew, step = self._step_lbfgs(GC, energy, iter_now, sub_iter_num=20)
            
            # Update GC and RGC
            GC = GC + step
            RGC = self._GC2RGC(GC)
            energy.append(energynew)
            
            iter_now += 1
            
        if iter_now >= max_iterations:
            print("\nReached maximum iterations without convergence")
            return False
            
        self.GC = GC
        return True

    def _step_newton(self, GC: torch.Tensor, dGC0: torch.Tensor, energy, iter_now, sub_iter_num = 10):
        """
        Perform a single step of the Newton-Raphson method.

        Args:
            GC (torch.Tensor): The current generalized coordinates.
            energy (list): List to store the energy values.
            iter_now (int): Current iteration number.
        Returns:
            energynew (float): The new energy value after the step.
            dGC (torch.Tensor): The change in generalized coordinates.
        """
        x_now = torch.zeros_like(GC)

        for i in range(sub_iter_num):
            RGC = self._GC2RGC(GC)
        
        # calculate the force vector and tangential stiffness matrix
        R, K_indices, K_values = self._assemble_Stiffness_Matrix(
            RGC=RGC)

        # solve the linear equation
        dGC = self._solve_linear_equation(K_indices=K_indices,
                                            K_values=K_values,
                                            R=-R,
                                            iter_now=iter_now,
                                            dGC0=dGC0).flatten()
        
        # backtracking line search to find the step size
        alpha, GCnew, energynew = self._line_search(
            GC, dGC, R, energy[-1])
        
        return energynew, dGC * alpha, alpha

    def _step_lbfgs(self, GC: torch.Tensor, energy, iter_now, sub_iter_num = 50):
        """
        solve the linear perturbation problem using L-BFGS method.

        Args:
            GC (torch.Tensor): The current generalized coordinates.
            energy (list): List to store the energy values.
            iter_now (int): Current iteration number.
            sub_iter_num (int, optional): Number of sub-iterations for L-BFGS. Defaults to 50.

        Returns:
            energynew (float): The new energy value after the step.
            dGC (torch.Tensor): The change in generalized coordinates.
        """

        x_now = torch.zeros_like(GC).requires_grad_()

        def closure(x_now: torch.Tensor):
            RGC = self._GC2RGC(GC+x_now)
            energynew = self._total_Potential_Energy(RGC=RGC)
            return energynew

        opt = _lbfgs.LBFGS(closure=closure, num_limit=500)
        for i in range(sub_iter_num):
            alpha, dk = opt.step(x_now)
            x_now.data += dk * alpha
            # print(f"LBFGS Iteration {i+1}/{sub_iter_num}, energy: {closure(x_now).item():.4e}\r", end='')

        # print()  # for a new line after the last iteration
        # print(f"Final energy after L-BFGS: {closure(x_now).item():.4e}")

        return closure(x_now).item(), x_now.detach()

    def _solve_linear_equation(self,
                               K_indices: torch.Tensor,
                               K_values: torch.Tensor,
                               R: torch.Tensor,
                               iter_now: int = 0,
                               alpha0: float = None,
                               dGC0: torch.Tensor = None,
                               tol_error=1e-8):
        if dGC0 is None:
            dGC0 = torch.zeros_like(R)

        if alpha0 is None:
            alpha0 = 1e-10

        # result = torch.sparse.spsolve(torch.sparse_coo_tensor(K_indices, K_values, [R.shape[0], R.shape[0]]).to_sparse_csr(), R)

        # precondition for the linear equation
        index = torch.where(K_indices[0] == K_indices[1])[0]
        diag = torch.zeros_like(R).scatter_add(0, K_indices[0, index],
                                               K_values[index]).sqrt()
        K_values_preconditioned = K_values / diag[K_indices[0]]
        K_values_preconditioned = K_values_preconditioned / diag[K_indices[1]]
        R_preconditioned = R / diag
        x0 = dGC0 * diag

        # record the number of low alpha
        if alpha0 < 1e-1:
            self.__low_alpha_count += 1
        else:
            self.__low_alpha_count = 0

        if self.__low_alpha_count > 3 or R_preconditioned.abs().max() < 1e-3:
            dx = _linear_Solver.pypardiso_solver(K_indices,
                                                 K_values_preconditioned,
                                                 R_preconditioned)
            self.__low_alpha_count = 0
        else:
            if iter_now % 20 == 0 or self.__low_alpha_count > 0:
                dx = _linear_Solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-4,
                                                       max_iter=6000)
            else:
                dx = _linear_Solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-4,
                                                       max_iter=2000)
        result = dx.to(R.dtype) / diag
        return result

    # endregion

    # region create CAD model

    def add_reference_point(self, rp: ReferencePoint, name: str = None):
        """
        Adds a reference point to the FEA object.

        Parameters:
            node (torch.Tensor): The node to be added as a reference point.

        Returns:
            str: The name of the reference point.
        """

        if name is None:
            number = len(self.reference_points)
            while ('rp-%d' % number) in self.reference_points:
                number += 1
            name = 'rp-%d' % number

        self.reference_points[name] = rp

        return name

    def delete_reference_point(self, name: str):
        """
        Deletes a reference point from the FEA object.

        Parameters:
        - name (str): The name of the reference point to be deleted.

        Returns:
        - None
        """
        if name in self.reference_points:
            del self.reference_points[name]
        else:
            raise ValueError(
                f"Reference point '{name}' not found in the model.")

    def add_load(self, load: loads.BaseLoad, name: str = None):
        """
        Add a load to the FEA model.

        Parameters:
            load (Load.Force_Base): The load to be added.

        Returns:
            str: The name of the load.
        """
        if name is None:
            number = len(self.loads)
            while ('load-%d' % number) in self.loads:
                number += 1
            name = 'load-%d' % number
        self.loads[name] = load

        return name

    def delete_load(self, name: str):
        """
        Delete a load from the FEA model.

        Parameters:
            name (str): The name of the load to be deleted.

        Returns:
            None
        """
        if name in self.loads:
            del self.loads[name]
        else:
            raise ValueError(f"Load '{name}' not found in the model.")

    def add_constraint(self,
                       constraint: constraints.BaseConstraint,
                       name: str = None):
        """
        Add a constraint to the FEA model.

        Parameters:
            constraint (Constraints.Constraints_Base): The constraint to be added.

        Returns:
            str: The name of the constraint.
        """
        if name is None:
            number = len(self.constraints)
            while ('constraint-%d' % number) in self.constraints:
                number += 1
            name = 'constraint-%d' % number
        self.constraints[name] = constraint
        return name

    def delete_constraint(self, name: str):
        """
        Delete a constraint from the FEA model.

        Parameters:
            name (str): The name of the constraint to be deleted.

        Returns:
            None
        """
        if name in self.constraints:
            del self.constraints[name]
        else:
            raise ValueError(f"Constraint '{name}' not found in the model.")

    def add_element(self, element: elements.BaseElement, name: str = None):
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

    def refine_RGC(self):
        for e in self.elems.values():
            e.refine_RGC(self.RGC, self.nodes)

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
            raise ValueError("At least two elements must be provided for merging")
            
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

    # endregion

    # region for IO
    def export_to_inp(self, filepath: str):
        """
        Export the model nodes and elements to an Abaqus INP file format.
        
        Args:
            filepath (str): Path to save the INP file.
            
        Returns:
            None
        """
        with open(filepath, 'w') as f:
            # Write header
            f.write('*Heading\n')
            f.write('Model exported from FEA_Main\n')
            
            # Write nodes
            f.write('*Node\n')
            nodes = self.nodes.cpu().numpy()
            for i in range(nodes.shape[0]):
                f.write(f"{i+1}, {nodes[i, 0]:.10e}, {nodes[i, 1]:.10e}, {nodes[i, 2]:.10e}\n")
            
            # Write elements by type
            element_counter = 1
            for elem_name, elem in self.elems.items():
                # Get element type name
                elem_type = elem.__class__.__name__
                elem_data = elem._elems.cpu().numpy()
                
                f.write(f"*Element, type={elem_type}\n")
                for e in range(elem_data.shape[0]):
                    # Convert 0-based indexing to 1-based for Abaqus
                    elem_nodes = [str(n + 1) for n in elem_data[e, :elem.num_nodes_per_elem]]
                    f.write(f"{element_counter}, {', '.join(elem_nodes)}\n")
                    element_counter += 1
                
                # Create element set for this element type
                f.write(f"*Elset, elset={elem_name}\n")
                elset_range = list(range(element_counter - elem_data.shape[0], element_counter))
                # Write in blocks of 16 (Abaqus standard)
                for i in range(0, len(elset_range), 16):
                    f.write(', '.join([str(j) for j in elset_range[i:i+16]]) + '\n')
            
            # Write any node sets if present
            for set_name, node_indices in self.node_sets.items():
                f.write(f"*Nset, nset={set_name}\n")
                # Convert 0-based indexing to 1-based for Abaqus
                node_indices = [n + 1 for n in node_indices]
                # Write in blocks of 16 (Abaqus standard)
                for i in range(0, len(node_indices), 16):
                    f.write(', '.join([str(j) for j in node_indices[i:i+16]]) + '\n')
                    
            # Write any element sets if present
            for set_name, elem_indices in self.element_sets.items():
                if set_name not in self.elems:  # Don't duplicate sets we already created
                    f.write(f"*Elset, elset={set_name}\n")
                    # Convert 0-based indexing to 1-based for Abaqus
                    elem_indices = [e + 1 for e in elem_indices]
                    # Write in blocks of 16 (Abaqus standard)
                    for i in range(0, len(elem_indices), 16):
                        f.write(', '.join([str(j) for j in elem_indices[i:i+16]]) + '\n')
                        
            print(f"Model exported to {filepath}")
    #endregion

    # region interface for GC
    def _allocate_RGC(self, size: list[int], name: str = None):
        """
        Allocate memory for the RGC data structure.

        Args:
        - size: A list of integers representing the size of the RGC tensor.
        - name: (optional) A string representing the name of the RGC tensor.

        Returns:
        None
        """

        index_now = max(list(self._RGC_size.keys())) + 1

        if name is None:
            name = 'RGC-%d' % index_now

        self._RGC_nameMap[index_now] = name
        self._RGC_size[index_now] = size

        self.RGC.append(torch.randn(size) * 1e-10)
        self.RGC_remain_index.append(np.zeros(size, dtype=bool))

        return index_now

    def _GC2RGC(self, GC: torch.Tensor):
        """
        Converts the global control vector (GC) to the reduced global control vector (RGC).

        Args:
            GC (torch.Tensor): The global control vector.

        Returns:
            list: The reduced global control vector (RGC).
        """
        RGC = []
        for i in range(len(self.RGC_remain_index)):
            RGC.append(torch.zeros(self._RGC_size[i]))
            RGC[-1][self.RGC_remain_index[i]] = GC[
                self._GC_list_indexStart[i]:self._GC_list_indexStart[i + 1]]

        for c in self.constraints.values():
            RGC = c.modify_RGC(RGC)

        return RGC

    def _GC2RGC_linear(self, GC: torch.Tensor):
        """
        Converts the global control vector (GC) to the reduced global control vector (RGC).

        Args:
            GC (torch.Tensor): The global control vector.

        Returns:
            list: The reduced global control vector (RGC).
        """
        RGC = []
        for i in range(len(self.RGC_remain_index)):
            RGC.append(torch.zeros(self._RGC_size[i]))
            RGC[-1][self.RGC_remain_index[i]] = GC[
                self._GC_list_indexStart[i]:self._GC_list_indexStart[i + 1]]

        for c in self.constraints.values():
            RGC = c.modify_RGC_linear(RGC)

        return RGC

    def _RGC2GC(self, RGC: list[torch.Tensor]):
        GC = torch.cat([
            RGC[i][self.RGC_remain_index[i]].flatten() for i in range(len(RGC))
        ],
                       dim=0)
        return GC

    # endregion

    # region for linear perturbation

    def solve_linear_perturbation(
        self,
        R0: torch.Tensor = None,
        R: torch.Tensor = None,
        GC0: torch.Tensor = torch.zeros([0])
    ) -> torch.Tensor:
        """
        Solve the linear perturbation problem.

        Args:
            R0 (torch.Tensor): The right-hand side vector.
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            torch.Tensor: The solution vector.
        """
        # initialize the RGC
        if GC0.numel() != 0:
            RGC = self._GC2RGC(GC0)
        else:
            RGC = self.RGC

        if R is None:
            if R0.dim() == 1:
                R0 = R0.unsqueeze(0)
            R = R0.clone()

            for c in self.constraints.values():
                for i in range(R.shape[0]):
                    R_new = c.modify_R(RGC, R0[i].flatten())
                    R[i] += R_new

            R = R[:, self.RGC_remain_index_flatten]

        if R.dim() == 1:
            R = R.unsqueeze(0)

        K_indices, K_values = self._assemble_Stiffness_Matrix(RGC=RGC)[1:]
        index = torch.where(K_indices[0] == K_indices[1])[0]
        diag = torch.zeros(R.shape[1]).scatter_add(0, K_indices[0, index],
                                                   torch.sqrt(K_values[index]))
        K_values_preconditioned = K_values / diag[K_indices[0]]
        K_values_preconditioned = K_values_preconditioned / diag[K_indices[1]]

        dGC = []
        for i in range(R.shape[0]):
            R_preconditioned = R[i] / diag
            dGC.append(
                _linear_Solver.conjugate_gradient(K_indices,
                                                  K_values_preconditioned,
                                                  R_preconditioned,
                                                  tol=1e-13,
                                                  max_iter=30000) / diag)
        dGC = torch.stack(dGC, dim=0)

        return dGC

    # endregion

    # region Sets Management
    def add_node_set(self, name: str, indices: np.ndarray):
        """
        Add a node set to the FEA model.
        
        Args:
            name (str): Name of the node set.
            indices (np.ndarray): Array of node indices.
            
        Returns:
            str: Name of the added node set.
        """
        self.node_sets[name] = np.array(indices, dtype=int)
        return name

    def add_element_set(self, name: str, indices: np.ndarray):
        """
        Add an element set to the FEA model.
        
        Args:
            name (str): Name of the element set.
            indices (np.ndarray): Array of element indices.
            
        Returns:
            str: Name of the added element set.
        """
        self.element_sets[name] = np.array(indices, dtype=int)
        return name

    def add_surface_set(self, name: str, elements: np.ndarray):
        """
        Add a surface set to the FEA model.
        
        Args:
            name (str): Name of the surface set.
            elements (np.ndarray): Surface elements information.
            
        Returns:
            str: Name of the added surface set.
        """
        self.surface_sets[name] = elements
        return name

    def get_node_set(self, name: str) -> np.ndarray:
        """
        Get a node set by name.
        
        Args:
            name (str): Name of the node set.
            
        Returns:
            np.ndarray: Array of node indices.
            
        Raises:
            KeyError: If the node set doesn't exist.
        """
        if name in self.node_sets:
            return self.node_sets[name]
        raise KeyError(f"Node set '{name}' not found in the model.")

    def get_element_set(self, name: str) -> np.ndarray:
        """
        Get an element set by name.
        
        Args:
            name (str): Name of the element set.
            
        Returns:
            np.ndarray: Array of element indices.
            
        Raises:
            KeyError: If the element set doesn't exist.
        """
        if name in self.element_sets:
            return self.element_sets[name]
        raise KeyError(f"Element set '{name}' not found in the model.")

    def get_surface_set(self, name: str) -> np.ndarray:
        """
        Get a surface set by name.
        
        Args:
            name (str): Name of the surface set.
            
        Returns:
            np.ndarray: Surface elements information.
            
        Raises:
            KeyError: If the surface set doesn't exist.
        """
        if name in self.surface_sets:
            return self.surface_sets[name]

        raise KeyError(f"Surface set '{name}' not found in the model.")

    def get_surface_elements(self, name: str) -> list[surfaces.BaseSurface]:
        """        Get the triangles of a surface set by name.  
        
        Args:
            name (str): Name of the surface set.
            
        Returns:
            list[BaseSurface]: List of triangles in the surface set.
            
        Raises:
            ValueError: If the surface set is not found.
        """
        surface = []
        for surf_index in self.surface_sets[name]:
            elem_ind = surf_index[0]
            surf_ind = surf_index[1]
            for e in self.elems.values():
                s_now = e.find_surface(surf_ind, elem_ind)
                if s_now is not None:
                    surface.append(s_now)
        if len(surface) == 0:
            raise ValueError(f"Surface {surf_ind} not found in the model.")
        else:
            return surfaces.merge_surfaces(surface)

    def delete_node_set(self, name: str):
        """
        Delete a node set from the FEA model.
        
        Args:
            name (str): Name of the node set to delete.
            
        Raises:
            KeyError: If the node set doesn't exist.
        """
        if name in self.node_sets:
            del self.node_sets[name]
        else:
            raise KeyError(f"Node set '{name}' not found in the model.")

    def delete_element_set(self, name: str):
        """
        Delete an element set from the FEA model.
        
        Args:
            name (str): Name of the element set to delete.
            
        Raises:
            KeyError: If the element set doesn't exist.
        """
        if name in self.element_sets:
            del self.element_sets[name]
        else:
            raise KeyError(f"Element set '{name}' not found in the model.")

    def delete_surface_set(self, name: str):
        """
        Delete a surface set from the FEA model.
        
        Args:
            name (str): Name of the surface set to delete.
            
        Raises:
            KeyError: If the surface set doesn't exist.
        """
        if name in self.surface_sets:
            del self.surface_sets[name]
        else:
            raise KeyError(f"Surface set '{name}' not found in the model.")

    # endregion

    # region for visualization

    def show_surface(self, name: list[str] = None, show: bool = True):
        """
        Show the surface of the FEA model.

        Args:
            name (list[str], optional): List of surface names to show. Defaults to None.
            color (str, optional): Color of the surface. Defaults to 'blue'.
            show (bool, optional): Whether to show the surface. Defaults to True.
        """
        from mayavi import mlab
        if name is None:
            name = list(self.surface_sets.keys())

        for n in name:
            if n in self.surface_sets:
                surface = self.get_surface_triangles(n)
                surface = torch.cat(surface, dim=0).cpu().numpy()
                mlab.triangular_mesh(self.nodes[:, 0].cpu().numpy(),
                                     self.nodes[:, 1].cpu().numpy(),
                                     self.nodes[:, 2].cpu().numpy(),
                                     surface,
                                     opacity=0.5)

                surface = mlab.pipeline.surface(
                    mlab.pipeline.triangular_mesh_source(
                        self.nodes[:, 0].cpu().numpy(),
                        self.nodes[:, 1].cpu().numpy(),
                        self.nodes[:, 2].cpu().numpy(), surface),
                    color=(1.0 / 255, 1.0 / 255, 1.0 / 255),
                    opacity=1)
                surface.actor.property.representation = 'wireframe'

        if show:
            mlab.show()

    # endregion
