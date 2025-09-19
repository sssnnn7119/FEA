from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Assembly
import time
import torch
from . import _linear_solver
from .basesolver import BaseSolver

class StaticImplicitSolver(BaseSolver):

    def __init__(self, maximum_iteration: int = 10000) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """

        self.maximum_iteration: int = maximum_iteration
        """
        the allowed maximum number of iterations for the solver.
        """

        self._iter_now: int = 0
        """
        The iteration of the FEA step
        """

        self._maximum_step_length = 1e10
        """
        The allowable maximum step length for each step.
        """

        self.__low_alpha_count = 0

        self.assembly: Assembly = None
        """ The assembly of the finite element model. """

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
        t0 = time.time()
        # start the iteration
        result = self._solve_iteration(RGC=RGC0 if RGC0 is not None else self.assembly.RGC,
                                         tol_error=tol_error)
        
        self.assembly.RGC = self.assembly.refine_RGC(self.assembly._GC2RGC(self.assembly.GC))
        t2 = time.time()

        # print the information
        print('total_iter:%d, total_time:%.2f' % (self._iter_now, t2 - t0))
        R = self.assembly.assemble_Stiffness_Matrix(RGC=self.assembly.RGC)[0]
        print('max_error:%.4e' % (R.abs().max()))
        print('---' * 8, 'FEA Finished', '---' * 8, '\n')

        return result
   

    # region solve iteration

    def _line_search(self,
                     GC0: torch.Tensor,
                     dGC: torch.Tensor,
                     R: torch.Tensor,
                     energy0: float, *args, **kwargs):
        # line search
        alpha = 1.0
        beta = float('inf')
        c1 = 0.3
        c2 = 0.4
        dGC0 = dGC.clone()
        deltaE = (dGC * R).sum()

        if deltaE > 0:
            dGC = -dGC
            deltaE = -deltaE
            print('the newton dirction is not the decrease direction')

        if torch.isnan(dGC).sum() > 0 or torch.isinf(dGC).sum() > 0:
            raise ValueError('dGC has nan or inf')
            dGC = -R
            deltaE = (dGC * R).sum()

        # if abs(deltaE / energy0) < tol_error:
        #     return 1, GC0

        loopc2 = 0
        while True:
            GCnew = GC0 + alpha * dGC
            # GCnew.requires_grad_()
            energy_new = self.assembly._total_Potential_Energy(
                RGC=self.assembly._GC2RGC(GCnew))

            if torch.isnan(energy_new) or torch.isinf(
                    energy_new) or \
                energy_new > energy0 + c1 * deltaE * alpha or \
                (alpha * dGC).abs().max() > self._maximum_step_length:
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

        # if abs(alpha) < 1e-6:
        #     # gradient direction line search
        #     alpha = 1
        #     dGC = R
        #     while True:
        #         GCnew = GC0 + alpha * dGC
        #         energy_new = self.assembly._total_Potential_Energy(
        #             RGC=self.assembly._GC2RGC(GCnew))
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

        GC = self.assembly._RGC2GC(RGC)
        RGC = self.assembly._GC2RGC(GC)

        # iteration now
        self._iter_now = 0

        # initialize the time
        t00 = time.time()

        # initialize the energy
        energy = [
            self.assembly._total_Potential_Energy(RGC=RGC)
        ]

        # check the initial energy, if nan, reinitialize the RGC
        if torch.isnan(energy[-1]):
            for i in range(len(self.assembly.RGC)):
                RGC[i] = torch.randn_like(self.assembly.RGC[i]) * 1e-10

            GC = self.assembly._RGC2GC(RGC)
            energy[-1] = self.assembly._total_Potential_Energy(
                RGC=RGC)
        dGC = torch.zeros_like(GC)

        # record the number of low alpha
        low_alpha = 0
        alpha = 0

        # begin the iteration
        while True:

            if self._iter_now > self.maximum_iteration:
                print('maximum iteration reached')
                self.assembly.GC = GC
                return False

            # calculate the force vector and tangential stiffness matrix
            t1 = time.time()
            R, K_indices, K_values = self.assembly.assemble_Stiffness_Matrix(
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
            if R.abs().max() > 1e-3:
                alpha, GCnew, energynew = self._line_search(
                    GC, dGC, R, energy[-1])
            else:
                alpha = 1.
                GCnew = GC + dGC
                energynew = self.assembly._total_Potential_Energy(RGC = self.assembly._GC2RGC(GCnew))

            if alpha==0 and R.abs().max() > tol_error:
                self.assembly.GC = GC
                return False
            if alpha==0:
                break

            # if convergence has difficulty, reduce the load percentage
            if alpha < 0.01:
                low_alpha += 1
            else:
                low_alpha -= 5
                if low_alpha < 0:
                    low_alpha = 0

            if low_alpha > 10:
                if R.abs().max() < 1e-3:
                    print('low alpha, but convergence achieved')
                    self.assembly.GC = GC
                    break
                return False

            # update the GC
            GC = GCnew

            # update the RGC
            RGC = self.assembly._GC2RGC(GC)

            # self.show_surface(nodes=self.nodes+RGC[0])

            # update the energy
            energynew = self.assembly._total_Potential_Energy(
                RGC=RGC)
            energy.append(energynew)

            # reinitialize the objects
            # for e in self.elems.values():
            #     e.reinitialize(RGC=RGC)

            # for f in self.loads.values():
            #     f.reinitialize(RGC=RGC)

            # for c in self.constraints.values():
            #     c.reinitialize(RGC=RGC)

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
            
            if dGC.abs().max() < tol_error and R.abs().max() < tol_error:
                break
        self.assembly.GC = GC
        return True

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
                                               K_values[index]).abs().sqrt()
        diag[diag==0] = 1.0  # Avoid division by zero
        K_values_preconditioned = K_values / diag[K_indices[0]]
        K_values_preconditioned = K_values_preconditioned / diag[K_indices[1]]
        R_preconditioned = R / diag
        x0 = dGC0 * diag

        # record the number of low alpha
        if alpha0 < 1e-1:
            self.__low_alpha_count += 1
        else:
            self.__low_alpha_count = 0

        if self.__low_alpha_count > 3 or R_preconditioned.abs().max() < 1e-3 or K_values_preconditioned.device.type == 'cpu':
            dx = _linear_solver.pypardiso_solver(K_indices,
                                                 K_values_preconditioned,
                                                 R_preconditioned)
            self.__low_alpha_count = 0
        else:
            if iter_now % 20 == 0 or self.__low_alpha_count > 0:
                dx = _linear_solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=6000)
            else:
                dx = _linear_solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=1500)
        result = dx.to(R.dtype) / diag
        return result

    # endregion

