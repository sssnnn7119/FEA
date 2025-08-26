from turtle import color, pd
from unicodedata import numeric
import numpy as np
import torch
from .base import BaseLoad
from ..elements import BaseSurface

class ContactSelf(BaseLoad):
    """
    Class representing self-contact loads in the finite element model.
    """

    def __init__(self, surface_name: str, 
                 penalty_distance_g: float = 0.2,
                 penalty_factor_g: float = 0.8,
                 penalty_degree: int = 9,
                 penalty_threshold_h: float = 1.5,
                 penalty_ratio_h: float = 0.8,
                 penalty_start_f: int = -0.6,
                 penalty_end_f: float = -0.9,
                 ignore_max_distance: float = 6.,
                 ignore_min_normal: float = 1.2,
                 ignore_max_normal: float = 1.8):
        """
        Initialize the self-contact load.

        Args:
            surface_name (str): The name of the surface to apply the load on.
        """

        super().__init__()

        self.surface_name = surface_name
        """The name of the surface to apply the load on."""

        self._ignore_max_distance = ignore_max_distance
        """The maximum initial distance to ignore for self-contact."""

        self._ignore_min_normal = ignore_min_normal
        """The minimum initial normal distance to ignore for self-contact."""
        self._ignore_max_normal = ignore_max_normal
        """The maximum initial normal distance to ignore for self-contact."""

        self._penalty_distance_g = penalty_distance_g
        """The penalty distance for self-contact. When the distance between nodes is less than this value, a penalty is applied."""

        self._penalty_factor_g = penalty_factor_g
        """The penalty factor g for self-contact."""

        self._penalty_threshold_h = penalty_threshold_h
        """The penalty threshold for self-contact."""

        self._penalty_ratio_h = penalty_ratio_h
        """The penalty ratio for self-contact."""

        self._penalty_start_f = penalty_start_f
        """The penalty degree for the angle factor f. The degree of the penalty function."""

        self._penalty_end_f = penalty_end_f
        """The penalty threshold for the angle factor f."""

        self._penalty_degree = penalty_degree
        """The penalty degree for self-contact. The degree of the penalty function."""

        self.surface_element: BaseSurface
        """The list of surface elements for self-contact."""

        self._gaussian_points: torch.Tensor
        """The Gaussian points for integration over the surface."""

        self._gaussian_weights_int: torch.Tensor
        """The Gaussian weights for integration over the surface."""

        self._points_surface_count: list[int]
        """The number of gaussian points on each surface element."""

        self._ratio: torch.Tensor
        """The ratio for self-contact to avoid the calculation of the nearest distance."""

        self._point_pairs: torch.Tensor
        """The point pairs that need to be considered for self-contact."""

        self._pdU_2_indices: torch.Tensor
        """The indices for the partial derivatives with respect to the second variable."""



    def initialize(self, fea):
        
        super().initialize(fea)

        # filter the point pairs
        self.surface_element = fea.surface_sets.get_elements(self.surface_name)[0]

        def filter_the_point_pairs(surface_element: BaseSurface, nodes: torch.Tensor):
            elems_gaussian = surface_element.gaussian_points_position(nodes)
            elems_mid = elems_gaussian.mean(dim=0).cpu()
            dy = elems_mid[:, None, :] - elems_mid[None, :, :]

            distance0 = dy.norm(dim=-1)
            index_remain = distance0 < self._ignore_max_distance
            index_remain.fill_diagonal_(False)
            index_remain = torch.triu(index_remain, diagonal=-1)
            point_pairs = torch.stack(torch.where(index_remain), dim=0).to(nodes.device)

            normal0 = surface_element.get_gaussian_normal(fea.nodes)
            normal0 = normal0 / normal0.norm(dim=-1, keepdim=True)

            dm = normal0[:, None, point_pairs[0], :] - normal0[None, :, point_pairs[1], :]
            dy = elems_gaussian[:, None, point_pairs[0], :] - elems_gaussian[None, :, point_pairs[1], :]
            dr = dy / dy.norm(dim=-1, keepdim=True)

            ratio_d = self._ratio_d_func(dx = dr, dm=dm)
            index_remain = (ratio_d.sum([0, 1]) > 0)

            point_pairs = point_pairs[:, index_remain]
            ratio_d = ratio_d[:, :, index_remain]

            # point_pairs = point_pairs[:, [0]]
            # ratio_d = ratio_d[:, :, [0]]
            return point_pairs, ratio_d

        self._point_pairs, self._ratio = filter_the_point_pairs(self.surface_element, fea.nodes)

        self._gaussian_weights_int = torch.einsum('ge, g->ge', self.surface_element.det_Jacobian, self.surface_element.gaussian_weight)
        self._gaussian_points = self.surface_element.gaussian_points_position(fea.nodes)

        self._ratio = self._ratio * self._gaussian_weights_int[:, None, self._point_pairs[0]] * self._gaussian_weights_int[None, :, self._point_pairs[1]]



        # get the partial derivative indices
        tri_ind = self._point_pairs.cpu()

        num_p = self._point_pairs.shape[1]
        num_n = self.surface_element.num_nodes_per_elem

        pdU_indices = self.surface_element._elems[tri_ind].to(torch.int64)
        pdU_indices = torch.stack([pdU_indices*3, pdU_indices*3+1, pdU_indices*3+2], dim=-1)
        self._pdU_indices = pdU_indices.to(self._fea.nodes.device)


        
        # self._pdU_2_indices = pdU_2_indices.to(self._fea.nodes.device)

    def _overlap_check(self, dy: torch.Tensor, dn: torch.Tensor):
        distance = dy.norm(dim=-1)
        T = -(dy*dn).sum(-1)

        check = ((T<0) & (distance<0.4)).sum()

        return check>0

    def get_potential_energy(self, RGC):
        
        U = RGC[0]
        Y = self._fea.nodes + U
        y = self.surface_element.gaussian_points_position(Y)
        N = self.surface_element.get_gaussian_normal(Y)

        nnorm = N.norm(dim=-1)
        n = N / nnorm[:, :, None]

        dy = y[:, None, self._point_pairs[0], :] - y[None, :, self._point_pairs[1], :]


        dn = n[:, None, self._point_pairs[0], :] - n[None, :, self._point_pairs[1], :]
        ndot = (n[:, None, self._point_pairs[0], :] * n[None, :, self._point_pairs[1], :]).sum(dim=-1)

        # if self._overlap_check(dy=dy,dn=dn):
        #     return torch.nan

        # distance_now = -(dn*dy).sum(dim=-1)
        MM = (self._penalty_start_f - ndot) / (self._penalty_start_f-self._penalty_end_f)
        MM = MM.clamp(0, 1)
        f = MM**3 * (6*MM**2 - 15*MM + 10)
        
        D = self._penalty_distance_g + (dn * dy).sum(dim=-1)
        D[D < 0] = 0
        g = (D / self._penalty_factor_g) ** self._penalty_degree


        L = dy.norm(dim=-1)
        T = (self._penalty_threshold_h - L) / (self._penalty_ratio_h * self._penalty_threshold_h)
        T = T.clamp(0, 1)
        h = T**3 * (6*T**2 - 15*T + 10)
        penalty = g * f * h

        # Compute the potential energy
        potential_energy = penalty.sum()

        return -potential_energy


    def get_stiffness(self, RGC) -> float:

        U = RGC[0]
        # U = U.detach().clone().requires_grad_()
        Y = self._fea.nodes + U

        num_g = self.surface_element._num_gaussian
        
        num_e = self.surface_element._elems.shape[0]
        num_n = self.surface_element.num_nodes_per_elem

        Ye = Y[self.surface_element._elems]

        y = torch.einsum('eai, ga->gei', Ye, self.surface_element.shape_function_gaussian[0])
        NR = torch.einsum('gma, eai->gemi', self.surface_element.shape_function_gaussian[1], Ye)
        N = torch.cross(NR[:, :, 0, :], NR[:, :, 1, :], dim=-1)

        nnorm = N.norm(dim=-1)
        n = N / nnorm[:, :, None]

        num_p = self._point_pairs.shape[1]
        E = torch.zeros([num_g, num_p, 2, 2, 3], device=U.device) # e1/e2, y/n, 0/1/2

        E[:, :, 0, 0] = y[:, self._point_pairs[0]]
        E[:, :, 1, 0] = y[:, self._point_pairs[1]]
        E[:, :, 0, 1] = n[:, self._point_pairs[0]]
        E[:, :, 1, 1] = n[:, self._point_pairs[1]]

        dy = E[:, None, :, 0, 0, :] - E[None, :, :, 1, 0, :]
        dn = E[:, None, :, 0, 1, :] - E[None, :, :, 1, 1, :]

        M = (E[:, None, :, 0, 1, :] * E[None, :, :, 1, 1, :]).sum(dim=-1)
        MM = (self._penalty_start_f - M) / (self._penalty_start_f-self._penalty_end_f)
        MM = MM.clamp(0, 1)
        f = MM**3 * (6*MM**2 - 15*MM + 10)

        D = self._penalty_distance_g + (dn * dy).sum(dim=-1)
        D[D < 0] = 0
        g = (D / self._penalty_factor_g) ** self._penalty_degree
        L = dy.norm(dim=-1)
        T = (self._penalty_threshold_h - L) / (self._penalty_ratio_h * self._penalty_threshold_h)
        T = T.clamp(0, 1)
        h = T**3 * (6*T**2 - 15*T + 10)

        penalty = g * f * h

        index_remain = torch.where(penalty.sum([0,1])>1e-12)[0]

        point_pairs = self._point_pairs[:, index_remain]
        num_p = index_remain.shape[0]
        D = D[:, :, index_remain]
        M = M[:, :, index_remain]
        MM = MM[:, :, index_remain]
        E = E[:, index_remain]
        T = T[:, :, index_remain]
        L = L[:, :, index_remain]
        dy = dy[:, :, index_remain]
        dn = dn[:, :, index_remain]
        f = f[:, :, index_remain]
        g = g[:, :, index_remain]
        h = h[:, :, index_remain]
        
        if index_remain.shape[0] > 0:
            print('  Contact pairs: ', index_remain.shape[0], '\t surface name: ', self.surface_name)
            # from mayavi import mlab
            # ind = 1100
            # point_pairs_show = point_pairs
            # mlab.figure()
            # mlab.triangular_mesh((self._fea.nodes+RGC[0]).cpu()[:, 0], (self._fea.nodes+RGC[0]).cpu()[:, 1], (self._fea.nodes+RGC[0]).cpu()[:, 2], self.surface_element._elems.cpu().numpy(), color=(0.5, 0.5, 0.5))
            # mlab.points3d(y[0][point_pairs_show[0], 0].cpu(), y[0][point_pairs_show[0], 1].cpu(), y[0][point_pairs_show[0], 2].cpu(), color=(1, 0, 0), scale_factor = 0.2)
            # mlab.points3d(y[0][point_pairs_show[1], 0].cpu(), y[0][point_pairs_show[1], 1].cpu(), y[0][point_pairs_show[1], 2].cpu(), color=(0, 0, 1), scale_factor = 0.2)

            # mlab.quiver3d(y[0][point_pairs_show[0], 0].cpu(), y[0][point_pairs_show[0], 1].cpu(), y[0][point_pairs_show[0], 2].cpu(), n[0][point_pairs_show[0], 0].cpu(), n[0][point_pairs_show[0], 1].cpu(), n[0][point_pairs_show[0], 2].cpu(), scale_factor=10.)
            # mlab.quiver3d(y[0][point_pairs_show[1], 0].cpu(), y[0][point_pairs_show[1], 1].cpu(), y[0][point_pairs_show[1], 2].cpu(), n[0][point_pairs_show[1], 0].cpu(), n[0][point_pairs_show[1], 1].cpu(), n[0][point_pairs_show[1], 2].cpu(), scale_factor=10.)
            # mlab.show()

        # Compute the potential energy
        potential_energy = penalty.sum()

        ndN = torch.einsum('ij, ge->geij', torch.eye(3), 1/nnorm) + \
            torch.einsum('gei, gej, ge->geij', n, n, -1/nnorm)
        ndN_2 = torch.einsum('ij, gek, ge->geijk', torch.eye(3), n, -1/nnorm**2) + \
            torch.einsum('geik, gej, ge->geijk', ndN, n, -1/nnorm) + \
            torch.einsum('gei, gejk, ge->geijk', n, ndN, -1/nnorm) + \
            torch.einsum('gei, gej, gek, ge->geijk', n, n, n, 1/nnorm**2)
        
        ydUe = self.surface_element.shape_function_gaussian[0]

        epsilon = torch.zeros([3, 3, 3])
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[1, 0, 2] = epsilon[2, 1, 0] = epsilon[0, 2, 1] = -1

        NR = torch.einsum('gma, eai->gemi', self.surface_element.shape_function_gaussian[1], Y[self.surface_element._elems])
        NdUe = torch.einsum('ijl, geja->geial', 
                            epsilon, 
                            torch.einsum('gei, ga->geia', NR[:, :, 0], self.surface_element.shape_function_gaussian[1][:, 1]) - 
                            torch.einsum('gei, ga->geia', NR[:, :, 1], self.surface_element.shape_function_gaussian[1][:, 0]))
        NdUe_2 = torch.einsum('ipl, gab->gialbp', epsilon, 
                              torch.einsum('gb,ga->gab', self.surface_element.shape_function_gaussian[1][:, 0], self.surface_element.shape_function_gaussian[1][:, 1])-
                              torch.einsum('gb,ga->gab', self.surface_element.shape_function_gaussian[1][:, 1], self.surface_element.shape_function_gaussian[1][:, 0]))

        ndUe = torch.einsum('geij, geial->gejal', ndN, NdUe)

        ndUe_2 = torch.einsum('geijk, geial, gekbp->gejalbp', ndN_2, NdUe, NdUe) + \
                torch.einsum('geij, gialbp->gejalbp', ndN, NdUe_2)

        edUe = torch.zeros([num_g, num_e, 2, 3, num_n, 3])
        edUe[:, :, 1] = ndUe
        edUe[:, :, 0, 0, :, 0] = ydUe[:, None, :]
        edUe[:, :, 0, 1, :, 1] = ydUe[:, None, :]
        edUe[:, :, 0, 2, :, 2] = ydUe[:, None, :]

        edUe_2 = torch.zeros([num_g, num_e, 2, 3, num_n, 3, num_n, 3])
        edUe_2[:, :, 1] = ndUe_2

        gdD = self._penalty_degree / self._penalty_factor_g * (D / self._penalty_factor_g) ** (self._penalty_degree-1)
        gdD_2 = self._penalty_degree * (self._penalty_degree - 1) / self._penalty_factor_g ** 2 * (D / self._penalty_factor_g) ** (self._penalty_degree - 2)

        gdE = torch.zeros([num_g, num_g, num_p, 2, 2, 3])
        gdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', gdD, dn)
        gdE[:, :, :, 1, 0, :] = -gdE[:, :, :, 0, 0, :]
        gdE[:, :, :, 0, 1, :] = torch.einsum('gGp, gGpi->gGpi', gdD, dy)
        gdE[:, :, :, 1, 1, :] = -gdE[:, :, :, 0, 1, :]

        gdE_2 = torch.zeros([num_g, num_g, num_p, 2, 2, 3, 2, 2, 3])
        tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2, dn, dn)
        gdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
        gdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
        gdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
        gdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

        tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2, dn, dy) + \
                torch.einsum('gGp, ij->gGpij', gdD, torch.eye(3))
        gdE_2[:, :, :, 0, 0, :, 0, 1, :] = tmp
        gdE_2[:, :, :, 0, 0, :, 1, 1, :] = -tmp
        gdE_2[:, :, :, 1, 0, :, 0, 1, :] = -tmp
        gdE_2[:, :, :, 1, 0, :, 1, 1, :] = tmp

        tmp = tmp.permute([0, 1, 2, 4, 3])
        gdE_2[:, :, :, 0, 1, :, 0, 0, :] = tmp
        gdE_2[:, :, :, 0, 1, :, 1, 0, :] = -tmp
        gdE_2[:, :, :, 1, 1, :, 0, 0, :] = -tmp
        gdE_2[:, :, :, 1, 1, :, 1, 0, :] = tmp

        temp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2, dy, dy)
        gdE_2[:, :, :, 0, 1, :, 0, 1, :] = temp
        gdE_2[:, :, :, 1, 1, :, 0, 1, :] = -temp
        gdE_2[:, :, :, 0, 1, :, 1, 1, :] = -temp
        gdE_2[:, :, :, 1, 1, :, 1, 1, :] = temp

        # MM = (self._penalty_start_f - M) / (self._penalty_start_f-self._penalty_end_f)
        # MM = MM.clamp(0, 1)
        # f = MM**3 * (6*MM**2 - 15*MM + 10)
        fdM = -30*MM**2*(MM-1)**2 / (self._penalty_start_f-self._penalty_end_f)
        fdM_2 = 60*MM*(MM-1)*(2*MM-1) / (self._penalty_start_f-self._penalty_end_f)**2
        fdM[MM>=1] = 0 
        fdM[MM<=0] = 0
        fdM_2[MM>=1] = 0 
        fdM_2[MM<=0] = 0
        # M = (E[:, None, :, 0, 1, :] * E[None, :, :, 1, 1, :]).sum(dim=-1)

        fdE = torch.zeros([num_g, num_g, num_p, 2, 2, 3])
        fdE[:, :, :, 0, 1, :] = torch.einsum('gGp, gGpi->gGpi', fdM, E[:, None, :, 1, 1, :])
        fdE[:, :, :, 1, 1, :] = torch.einsum('gGp, gGpi->gGpi', fdM, E[None, :, :, 0, 1, :])

        fdE_2 = torch.zeros([num_g, num_g, num_p, 2, 2, 3, 2, 2, 3])
        fdE_2[:, :, :, 0, 1, :, 0, 1, :] = torch.einsum('gGp, gGpi, gGpj->gGpij', fdM_2, E[:, None, :, 1, 1, :], E[:, None, :, 1, 1, :])
        fdE_2[:, :, :, 0, 1, :, 1, 1, :] = torch.einsum('gGp, ij->gGpij', fdM, torch.eye(3)) + \
                                            torch.einsum('gGp, gGpi, gGpj->gGpij', fdM_2, E[:, None, :, 1, 1, :], E[:, None, :, 0, 1, :])
        fdE_2[:, :, :, 1, 1, :, 1, 1, :] = torch.einsum('gGp, gGpi, gGpj->gGpij', fdM_2, E[:, None, :, 0, 1, :], E[:, None, :, 0, 1, :])
        fdE_2[:, :, :, 1, 1, :, 0, 1, :] = torch.einsum('gGp, ij->gGpij', fdM, torch.eye(3)) + \
        torch.einsum('gGp, gGpi, gGpj->gGpij', fdM_2, E[:, None, :, 0, 1, :], E[:, None, :, 1, 1, :])

        hdE = torch.zeros([num_g, num_g, num_p, 2, 2, 3])

        # L = dy.norm(dim=-1)
        # T = (self._penalty_distance - L) / (0.5 * self._penalty_distance)
        # T = T.clamp(0, 1)
        # h = T**3 * (6*T**2 - 15*T + 10)
        Lddy = torch.einsum('gGpi, gGp->gGpi', dy, 1/L)
        Lddy_2 = torch.einsum('ij, gGp->gGpij', torch.eye(3), 1/L) + torch.einsum('gGpi, gGpj, gGp->gGpij', dy, Lddy, -1/L**2)
        hdL = -30*T**2*(T-1)**2 / (self._penalty_ratio_h * self._penalty_threshold_h)
        hdL_2 = 60*T*(T-1)*(2*T-1) / (self._penalty_ratio_h * self._penalty_threshold_h)**2
        hdL[T>=1] = 0
        hdL[T<=0] = 0
        hdL_2[T>=1] = 0
        hdL_2[T<=0] = 0
        hdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', hdL, Lddy)
        hdE[:, :, :, 1, 0, :] = -hdE[:, :, :, 0, 0, :]

        hdE_2 = torch.zeros([num_g, num_g, num_p, 2, 2, 3, 2, 2, 3])
        tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', hdL_2, Lddy, Lddy) + \
                torch.einsum('gGp, gGpij->gGpij', hdL, Lddy_2)
        hdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
        hdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
        hdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
        hdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

        pdE = torch.einsum('gGpmxi, gGp, gGp->gGpmxi', fdE, g, h) + \
            torch.einsum('gGp, gGpmxi, gGp->gGpmxi', f, gdE, h) + \
            torch.einsum('gGp, gGp, gGpmxi->gGpmxi', f, g, hdE)

        pdE_2 = torch.einsum('gGpmxinyj, gGp, gGp->gGpmxinyj', fdE_2, g, h) + \
                torch.einsum('gGpmxi, gGpnyj, gGp->gGpmxinyj', fdE, gdE, h) + \
                torch.einsum('gGpmxi, gGp, gGpnyj->gGpmxinyj', fdE, g, hdE) + \
                \
                torch.einsum('gGpnyj, gGpmxi, gGp->gGpmxinyj', fdE, gdE, h) +\
                torch.einsum('gGp, gGpmxinyj, gGp->gGpmxinyj', f, gdE_2, h) +\
                torch.einsum('gGp, gGpmxi, gGpnyj->gGpmxinyj', f, gdE, hdE) +\
                \
                torch.einsum('gGpnyj, gGp, gGpmxi->gGpmxinyj', fdE, g, hdE)+\
                torch.einsum('gGp, gGpnyj, gGpmxi->gGpmxinyj', f, gdE, hdE)+\
                torch.einsum('gGp, gGp, gGpmxinyj->gGpmxinyj', f, g, hdE_2)
        
 
        # pdUe = torch.zeros([num_e, num_n, 3])
        pdEsum0 = pdE.sum(0)
        pdEsum1 = pdE.sum(1)
        pdUe_values0 = torch.einsum('gpxi, gpxial->pal', pdEsum1[:, :, 0], edUe[:, point_pairs[0]])
        pdUe_values1 = torch.einsum('gpxi, gpxial->pal', pdEsum0[:, :, 1], edUe[:, point_pairs[1]])

        # for i in range(point_pairs.shape[1]):
        #     pdUe[point_pairs[0, i]] += pdUe_values0[i]
        #     pdUe[point_pairs[1, i]] += pdUe_values1[i]

        pdU_values = torch.stack([pdUe_values0, pdUe_values1], dim=0)
        

        # pdU = torch.zeros_like(Y).flatten().scatter_add_(0, pdU_indices.flatten(), pdU_values.flatten()).reshape([-1, 3])

        pdUe_2_values00 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(1)[:, :, 0, :, :, 0], edUe[:, point_pairs[0]], edUe[:, point_pairs[0]]) + \
                            torch.einsum('gpxi, gpxialbL->palbL', pdEsum1[:, :, 0], edUe_2[:, point_pairs[0]])
        
        pdUe_2_values01 = torch.einsum('gGpxiyj, gpxial, GpyjbL->palbL', pdE_2[:, :, :, 0, :, :, 1], edUe[:, point_pairs[0]], edUe[:, point_pairs[1]])

        pdUe_2_values10 = torch.einsum('gGpxiyj, Gpxial, gpyjbL->palbL', pdE_2[:, :, :, 1, :, :, 0], edUe[:, point_pairs[1]], edUe[:, point_pairs[0]])
        
        pdUe_2_values11 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(0)[:, :, 1, :, :, 1], edUe[:, point_pairs[1]], edUe[:, point_pairs[1]]) + \
                            torch.einsum('gpxi, gpxialbL->palbL', pdEsum0[:, :, 1], edUe_2[:, point_pairs[1]])

        pdU_2_values = torch.stack([pdUe_2_values00, pdUe_2_values01, pdUe_2_values10, pdUe_2_values11], dim=0)


        # pdU_2 = torch.sparse_coo_tensor(pdU_2_indices, pdU_2_values.flatten(), size=Y.shape*2)
        tri_ind = point_pairs.cpu()

        pdU_indices = self.surface_element._elems[tri_ind].to(torch.int64)
        pdU_indices = torch.stack([pdU_indices*3, pdU_indices*3+1, pdU_indices*3+2], dim=-1)
        pdU_indices = pdU_indices.to(self._fea.nodes.device)

        pdU_2_indices00 = torch.stack([
            self.surface_element._elems[tri_ind[0]].reshape([num_p, num_n, 1, 1, 1]).repeat([1, 1, 3, num_n, 3]),
            torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n, 1, num_n, 3]),
            self.surface_element._elems[tri_ind[0]].reshape([num_p, 1, 1, num_n, 1]).repeat([1, num_n, 3, 1, 3]),
            torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n, 3, num_n, 1]),
        ])

        pdU_2_indices01 = torch.stack([
            self.surface_element._elems[tri_ind[0]].reshape([num_p, num_n, 1, 1, 1]).repeat([1, 1, 3, num_n, 3]),
            torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n, 1, num_n, 3]),
            self.surface_element._elems[tri_ind[1]].reshape([num_p, 1, 1, num_n, 1]).repeat([1, num_n, 3, 1, 3]),
            torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n, 3, num_n, 1]),
        ])

        pdU_2_indices10 = torch.stack([
            self.surface_element._elems[tri_ind[1]].reshape([num_p, num_n, 1, 1, 1]).repeat([1, 1, 3, num_n, 3]),
            torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n, 1, num_n, 3]),
            self.surface_element._elems[tri_ind[0]].reshape([num_p, 1, 1, num_n, 1]).repeat([1, num_n, 3, 1, 3]),
            torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n, 3, num_n, 1]),
        ])

        pdU_2_indices11 = torch.stack([
            self.surface_element._elems[tri_ind[1]].reshape([num_p, num_n, 1, 1, 1]).repeat([1, 1, 3, num_n, 3]),
            torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n, 1, num_n, 3]),
            self.surface_element._elems[tri_ind[1]].reshape([num_p, 1, 1, num_n, 1]).repeat([1, num_n, 3, 1, 3]),
            torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n, 3, num_n, 1]),
        ])

        pdU_2_indices_ = torch.stack([pdU_2_indices00, pdU_2_indices01, pdU_2_indices10, pdU_2_indices11], dim=1)
        pdU_2_indices = torch.stack([pdU_2_indices_[0]*3+pdU_2_indices_[1], pdU_2_indices_[2]*3+pdU_2_indices_[3]], dim=0).to(self._fea.nodes.device)

        return pdU_indices.flatten(), -pdU_values.flatten(), pdU_2_indices.reshape([2, -1]), -pdU_2_values.flatten()

    def _ratio_c_func(self, dx: torch.Tensor):
        """
        Compute the ratio function for the self-contact load.

        Args:
            dx (torch.Tensor): the distance between nodes.

        Returns:
            torch.Tensor: The computed ratio.
        """
        u = (dx - self._ignore_min_distance) / (self._ignore_max_distance - self._ignore_min_distance)
        u = u.clamp(0, 1)
        return 6 * u**5 - 15 * u**4 + 10 * u**3

    def _ratio_d_func(self, dx: torch.Tensor, dm: torch.Tensor):
        """
        Compute the ratio function for the self-contact load.

        Args:
            dx (torch.Tensor): the distance between nodes.
            dm (torch.Tensor): the distance between normals.

        Returns:
            torch.Tensor: The computed ratio.
        """

        dx = dx / dx.norm(dim=-1, keepdim=True)

        T = - (dm * dx).sum(-1)
        T = (T - self._ignore_min_normal) / (self._ignore_max_normal - self._ignore_min_normal)
        T = T.clamp(0, 1)
        return 6 * T**5 - 15 * T**4 + 10 * T**3

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[0][self.surface_element._elems.flatten().unique().cpu()] = True
        return RGC_remain_index