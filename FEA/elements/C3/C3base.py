


import pandas as pd
import torch
import numpy as np
from ..base import BaseElement


class Element_3D(BaseElement):

    def __init__(self, elems_index: torch.Tensor,
                 elems: torch.Tensor) -> None:
        super().__init__(elems_index, elems)

        self.shape_function_gaussian: list[torch.Tensor] = []
        """
            the shape functions of each guassian point
            [
                [
                    g: guassian point
                    e: element
                    a: a-th node
                ],
                [
                    g: guassian point
                    e: element
                    i: derivative
                    a: a-th node
                ]
            ]
        """

        self.shape_function: list[torch.Tensor]
        """
            the shape functions of the element

            # coordinates: (g,h,r) in the local coordinates
                0: constant,\n
                1: g,\n
                2: h,\n
                3: r,\n
                4: g*h,\n
                5: h*r,\n
                6: r*g,\n
                7: g^2,\n
                8: h^2,\n
                9: r^2,\n
                10: g^2*h,\n
                11: h^2*g,\n
                12: h^2*r,\n
                13: r^2*h,\n
                14: r^2*g,\n
                15: g^2*r,\n
                16: g*h*r,\n
                17: g^3,\n
                18: h^3,\n
                19: r^3,\n

            # the shape of shape_function 
                
                a-th func,\n
                b-th coordinates
                
            # its derivative:
                
                m-th derivative,\n
                a-th func,\n
                b-th coordinates
                
            # and its 2-nd derivative
                
                m-th derivative,\n
                n-th derivative,\n
                a-th func,\n
                b-th coordinates
                
        """
                
        self.order: int = 1
        """
            whether to reduce the order of the element
            if True, the element will be reduced to 4 nodes
            if False, the element will remain 10 nodes
        """

        self.order_faces: torch.Tensor 
        """            
        the order of the faces of the element
            shape: [surface, element]
            0: linear, 1: quadratic
        """

        self._num_gaussian: int
        """
            the number of guassian points
        """
    
    def initialize(self, fea) -> None:

        super().initialize(fea)
        # coo index of the stiffness matricx of structural stress
        indices0 = []
        for i in range(self.num_nodes_per_elem):
            for l1 in range(3):
                for j in range(self.num_nodes_per_elem):
                    for l2 in range(3):
                        # for k in range(num_elems):
                        index1 = self._elems[:, i] * 3 + torch.ones(
                            self._elems.shape[0],
                            dtype=torch.int64,
                            device='cpu') * l1
                        index3 = self._elems[:, j] * 3 + torch.ones(
                            self._elems.shape[0],
                            dtype=torch.int64,
                            device='cpu') * l2
                        indices0.append(torch.stack([index1, index3], dim=0))
        index0 = torch.cat(indices0, dim=1)
        
        # some trick to get the unique index and accelerate the calculation
        scaler = index0.max() + 1
        index1 = index0[0] * scaler + index0[1]
        index_sorted_matrix = index1.argsort()
        index2 = index1[index_sorted_matrix]
        index_unique, self._index_matrix_coalesce = torch.unique_consecutive(
            index2, return_inverse=True)

        inverse_index = torch.zeros_like(index_sorted_matrix,
                                         device='cpu',
                                         dtype=torch.int64)
        inverse_index[index_sorted_matrix] = torch.arange(
            0, index_sorted_matrix.max() + 1, device='cpu', dtype=torch.int64)

        default_device = torch.zeros([1]).device

        self._index_matrix_coalesce = self._index_matrix_coalesce[inverse_index].to(
            default_device)
        self._indices_matrix = torch.zeros([2, index_unique.shape[0]],
                                          dtype=torch.int64)
        self._indices_matrix[1] = index_unique % scaler
        self._indices_matrix[0] = index_unique // scaler

        # coo index of the force vector of structural stress
        self._indices_force = self._elems[:, :self.num_nodes_per_elem].transpose(0, 1).unsqueeze(1).repeat(
            1, 3, 1)
        self._indices_force *= 3
        self._indices_force[:, 1, :] += 1
        self._indices_force[:, 2, :] += 2
        self._indices_force = self._indices_force.flatten().to(default_device)

    def _pre_load_gaussian(self, gauss_coordinates: torch.Tensor, nodes: torch.Tensor):
        """
        load the guassian points and its weight

        Args:
            gauss_coordinates: [g, 3], the local coordinates of the element
            nodes: [p, 3], the global coordinates of the element
        """

        pp = torch.zeros([self._num_gaussian, self.shape_function[0].shape[1]])
        pp[:, 0] = 1
        pp[:, 1] = gauss_coordinates[:, 0]
        pp[:, 2] = gauss_coordinates[:, 1]
        pp[:, 3] = gauss_coordinates[:, 2]
        if self.shape_function[0].shape[1] > 4:
            pp[:, 4] = gauss_coordinates[:, 0] * gauss_coordinates[:, 1]
            pp[:, 5] = gauss_coordinates[:, 1] * gauss_coordinates[:, 2]
            pp[:, 6] = gauss_coordinates[:, 2] * gauss_coordinates[:, 0]
        if self.shape_function[0].shape[1] > 7:
            pp[:, 7] = gauss_coordinates[:, 0]**2
            pp[:, 8] = gauss_coordinates[:, 1]**2
            pp[:, 9] = gauss_coordinates[:, 2]**2
        if self.shape_function[0].shape[1] > 10:
            pp[:, 10] = gauss_coordinates[:, 0]**2 * gauss_coordinates[:, 1]
            pp[:, 11] = gauss_coordinates[:, 1]**2 * gauss_coordinates[:, 0]
            pp[:, 12] = gauss_coordinates[:, 1]**2 * gauss_coordinates[:, 2]
            pp[:, 13] = gauss_coordinates[:, 2]**2 * gauss_coordinates[:, 1]
            pp[:, 14] = gauss_coordinates[:, 2]**2 * gauss_coordinates[:, 0]
            pp[:, 15] = gauss_coordinates[:, 0]**2 * gauss_coordinates[:, 2]
            pp[:, 16] = gauss_coordinates[:, 0] * gauss_coordinates[:, 1] * \
                        gauss_coordinates[:, 2]
        if self.shape_function[0].shape[1] > 17:
            pp[:, 17] = gauss_coordinates[:, 0]**3
            pp[:, 18] = gauss_coordinates[:, 1]**3
            pp[:, 19] = gauss_coordinates[:, 2]**3

        Jacobian = torch.zeros([self._num_gaussian, len(self._elems), 3, 3])
        shape_now = self.shape_function[1]
        for i in range(self.num_nodes_per_elem):
            Jacobian += torch.einsum('gb,mb,ei->geim', pp, shape_now[:, i],
                                     nodes[self._elems[:, i]])

        # Jacobian_Function
        # J: g(Gaussian) * e * 3(ref) * 3(rest)
        det_Jacobian = Jacobian.det()
        inv_Jacobian = Jacobian.inverse()
        shapeFun1 = torch.einsum('gemi,gb,mab->geia', inv_Jacobian, pp,
                                shape_now)
        shapeFun0 = torch.einsum('ab, gb->ga', self.shape_function[0],
                                      pp)
        
        self.shape_function_gaussian = [shapeFun0, shapeFun1]
        self.gaussian_weight = torch.einsum('ge, g->ge', det_Jacobian, self.gaussian_weight)

    def _shape_function_derivative(self, shape_function: torch.Tensor, ind: int):
        """
        get the derivative of the shape function

        Args:
            shape_function: [i, m], the shape function of the element
            ind: the index of the derivative

        Returns:
            torch.Tensor: the derivative of the shape function
        """

        # (1,x,y,z,xy,yz,zx,xx,yy,zz)
        result = torch.zeros_like(shape_function)
        if ind == 0:
            result[:, 0] = shape_function[:, 1]
            if shape_function.shape[1] > 4:
                result[:, 2] = shape_function[:, 4]
                result[:, 3] = shape_function[:, 6]
            if shape_function.shape[1] > 7:
                result[:, 1] = 2 * shape_function[:, 7]
            if shape_function.shape[1] > 10:
                result[:, 4] = 2 * shape_function[:, 10]
                result[:, 8] = shape_function[:, 11]
                result[:, 9] = shape_function[:, 14]
                result[:, 6] = 2 * shape_function[:, 15]
                result[:, 5] = shape_function[:, 16]
            if shape_function.shape[1] > 17:
                result[:, 7] = 3 * shape_function[:, 17]

        if ind == 1:
            result[:, 0] = shape_function[:, 2]
            if shape_function.shape[1] > 4:
                result[:, 1] = shape_function[:, 4]
                result[:, 3] = shape_function[:, 5]
            if shape_function.shape[1] > 7:
                result[:, 2] = 2 * shape_function[:, 8]
            if shape_function.shape[1] > 10:
                result[:, 7] = shape_function[:, 10]
                result[:, 4] = 2 * shape_function[:, 11]
                result[:, 5] = 2 * shape_function[:, 12]
                result[:, 9] = shape_function[:, 13]
                result[:, 6] = shape_function[:, 16]
            if shape_function.shape[1] > 17:
                result[:, 8] = 3 * shape_function[:, 18]
                

        if ind == 2:
            result[:, 0] = shape_function[:, 3]
            if shape_function.shape[1] > 4:
                result[:, 1] = shape_function[:, 6]
                result[:, 2] = shape_function[:, 5]
            if shape_function.shape[1] > 7:
                result[:, 3] = 2 * shape_function[:, 9]
            if shape_function.shape[1] > 10:
                result[:, 8] = shape_function[:, 12]
                result[:, 5] = 2 * shape_function[:, 13]
                result[:, 6] = 2 * shape_function[:, 14]
                result[:, 7] = shape_function[:, 15]
                result[:, 4] = shape_function[:, 16]
            if shape_function.shape[1] > 17:
                result[:, 9] = 3 * shape_function[:, 19]

        return result
  
    def get_gaussian_points(self, nodes: torch.Tensor) -> torch.Tensor:
        """
            get the gaussian points of the element
        """
        points_request = torch.zeros(
            [self._num_gaussian, self._elems.shape[0], 3])
        for i in range(self._elems.shape[1]):
            points_request = points_request + torch.einsum(
                'g,eI->geI', self.shape_function_gaussian[0][:, i], nodes[self._elems[:,
                                                                         i]])
        return points_request
    
    def potential_Energy(self, RGC: list[torch.Tensor]):
        
        U = RGC[0].reshape([-1, 3])
        Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                         self.shape_function_gaussian[1][:, :, :, i],
                                         U[self._elems[:, i]])

        F = Ugrad.clone()
        F[:, :, 0, 0] += 1
        F[:, :, 1, 1] += 1
        F[:, :, 2, 2] += 1

        J = F.det()
        I1 = (F**2).sum([-1, -2]) * J**(-2 / 3)

        W = torch.zeros([self._num_gaussian, self._elems.shape[0]])
        W = self.materials.strain_energy_density_C3(F=F,)
        
        Ea = torch.einsum(
            'ge,ge->',W,
            self.gaussian_weight)

        return Ea

    def structural_Force(self, RGC: list[torch.Tensor]):
        
        U = RGC[0].reshape([-1, 3])
        DG, I1, J, invF, s, C = self.components_Solid(U=U)
        
        # calculate the element residual force
        Relement = torch.einsum('geij,geia,ge->aje', s,
                                self.shape_function_gaussian[1],
                                self.gaussian_weight).flatten()
        
        # calculate the element tangential stiffness matrix
        Ka_element = torch.einsum('geijkl,gelb,geia,ge->ajbke',
                                   C,
                                  self.shape_function_gaussian[1],
                                  self.shape_function_gaussian[1],
                                  self.gaussian_weight).flatten()
        
        # assembly the stiffness matrix and residual force                 
        
        ## stiffness matrix

        values = torch.zeros([self._indices_matrix.shape[1]]).scatter_add(0, self._index_matrix_coalesce, Ka_element)
        

        return self._indices_force, Relement, self._indices_matrix, values

    def components_Solid(self, U: torch.Tensor):
        Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                         self.shape_function_gaussian[1][:, :, :, i],
                                         U[self._elems[:, i]])

        F = Ugrad.clone()
        F[:, :, 0, 0] += 1
        F[:, :, 1, 1] += 1
        F[:, :, 2, 2] += 1

        invF = F.inverse()
        J = F.det()
        Jneg = J**(-2 / 3)
        I1 = (F**2).sum([-1, -2]) * Jneg
        
        s = torch.zeros_like(F)
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3])

        s, C = self.materials.material_Constitutive_C3(F=F,
                                                    J=J,
                                                    Jneg=Jneg,
                                                    invF=invF,
                                                    I1=I1)

        return F, I1, J, invF, s, C

    def get_volumn(self, U: torch.Tensor = None):
        if U is None:
            return self.gaussian_weight.sum()
        else:
            Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
            for i in range(self.num_nodes_per_elem):
                Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                            self.shape_function_gaussian[1][:, :, :, i],
                                            U[self._elems[:, i]])
            F = Ugrad.clone()
            F[:, :, 0, 0] += 1
            F[:, :, 1, 1] += 1
            F[:, :, 2, 2] += 1
            J = F.det()
 
            return (self.gaussian_weight * J).sum()

    def map(self, U: torch.Tensor, p0: torch.Tensor, derivative: int):
        """
        map the element to the global coordinate system
            
        Args:
            U: [P, 3], the global displacement of the element
            p0: [p, 3], the local coordinates of the element

            derivative: 0 for the shape function, 1 for its derivative

        Returns:
            torch.Tensor: the global coordinates of the element
        """
        pp = torch.zeros([p0.shape[0], 10])
        pp[:, 0] = 1
        pp[:, 1] = p0[:, 0]
        pp[:, 2] = p0[:, 1]
        pp[:, 3] = p0[:, 2]
        pp[:, 4] = p0[:, 0] * p0[:, 1]
        pp[:, 5] = p0[:, 1] * p0[:, 2]
        pp[:, 6] = p0[:, 2] * p0[:, 0]
        pp[:, 7] = p0[:, 0]**2
        pp[:, 8] = p0[:, 1]**2
        pp[:, 9] = p0[:, 2]**2
        shape_now = self.shape_function[derivative]

        results = torch.zeros([p0.shape[0], 3, 3])
        for i in range(self.num_nodes_per_elem):
            results += torch.einsum('pb,mb,ei->peim', pp, shape_now[:, i],
                                     U[self._elems[:, i]])
            
        return results

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[0][self._elems.unique()] = True
        return RGC_remain_index
    


    # region second order methods

    def get_2nd_order_point_index(self):
        """
        get the 2-nd order point index of the element that lies in the middle of the element
        
        Returns:
            torch.Tensor: the 2-nd order point index of the element \n
                [0]: the index of the middle node of the element\n
                [1]: the index of the neighbor node of the middle node of the element\n
                [2]: the index of the other neighbor node of the middle node of the element\n
        """
        return torch.zeros([0, 3], dtype=torch.int64, device='cpu')

    def refine_RGC(self, RGC: list[torch.Tensor], nodes: torch.Tensor) -> list[torch.Tensor]:
        """
        Refine Reference Grid Coordinates for mid-edge nodes
        
        Args:
            RGC: List of Reference Grid Coordinates
            nodes: Node coordinates
            
        Returns:
            Updated RGC
        """
        mid_nodes_index = self.get_2nd_order_point_index()
        RGC[0][mid_nodes_index[:, 0]] = (RGC[0][mid_nodes_index[:, 1]] + RGC[0][mid_nodes_index[:, 2]]) / 2 + (nodes[mid_nodes_index[:, 1]] + nodes[mid_nodes_index[:, 2]] - 2 * nodes[mid_nodes_index[:, 0]]) / 2
        
        return RGC
    
    def set_order(self, order: int) -> None:
        self.order = order
    

    # endregion second order methods