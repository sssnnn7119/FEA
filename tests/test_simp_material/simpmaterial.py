from typing import Optional

import torch

torch.set_default_device('cpu')
torch.set_default_dtype(torch.float64)

import torchfea

class SIMPElement(torchfea.elements.Element_3D):

    def __init__(self, elems_index, elems, penalfactor: torch.Tensor):
        super().__init__(elems_index, elems)

        self.penalfactor = penalfactor
        """the penalization factor for SIMP material"""

    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)
        
        Ea = super().potential_Energy(RGC, rotation_matrix)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('gkij,kI->gkIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
            
        Em = self.penalfactor * torch.einsum('gkij,gkij->gk', Ugrad2, Ugrad2)


        return Ea + Em
    
    def _get_EpdUe_EpdUe2(self, U, if_onlyforce = False):
        result0 = super()._get_EpdUe_EpdUe2(U, if_onlyforce)
        
        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,kI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])

        EmdUgrad2 = 2 * self.penalfactor * Ugrad2

        dN2W = torch.einsum('gkija,ge->gkija', self.shape_function_d2_gaussian, self.gaussian_weight)

        EmdUe = torch.einsum('geIij,geija->aIe', EmdUgrad2,
                                self._dNW)
        
        if if_onlyforce:
            return EmdUe + result0
        
        EmdUe2 = torch.zeros([1, 1, 3, 3, 3, 3, 3, 3])
        for I0 in range(3):
            for i0 in range(3):
                for j0 in range(3):
                    for I1 in range(3):
                        for i1 in range(3):
                            for j1 in range(3):
                                EmdUe2[..., I0, i0, j0, I1, i1, j1] = self.penalfactor

        dN2dN2W = torch.einsum('gkija,gkklb->gkijaklb', dN2W, self.shape_function_d2_gaussian)

        EmdUe_2 = torch.einsum('gkijaklb,geIijJkl->aIbJe', dN2dN2W, EmdUe2)

        return EmdUe + result0[0], EmdUe_2 + result0[1]

        
    

if __name__ == '__main__':
    fe = torchfea.load_model('tests/test_simp_material/simpmodel.npz')

    fe.assembly.get_load('pressure_1').pressure = 0.01

    part = fe.assembly.get_part('final_model')

    elem_c3d4 = part.elems['C3D4']

    fe.initialize()

    GC0 = torch.zeros([fe.assembly.GC.shape[0]])



    raise False