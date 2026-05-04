from typing import Optional

import torch

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)

import torchfea

class SIMPElement(torchfea.elements.Element_3D):

    def __init__(self, elems_index, elems, penalfactor: torch.Tensor):
        super().__init__(elems_index, elems)

        self.penalfactor = penalfactor
        """the penalization factor for SIMP material"""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._dN2W = torch.einsum('geija,ge->geija', self.shape_function_d2_gaussian, self.gaussian_weight)

        EmdUgrad2_2 = torch.zeros([1, 1, 3, 3, 3, 3, 3, 3])
        for I0 in range(3):
            for i0 in range(3):
                for j0 in range(3):
                    EmdUgrad2_2[..., I0, i0, j0, I0, i0, j0] = self.penalfactor * 2

        self._EmdUe_2 = torch.einsum('geija, geklb,geIijJkl->aIbJe', self._dN2W, self.shape_function_d2_gaussian, EmdUgrad2_2)

    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)
        
        Ea = super().potential_Energy(RGC, rotation_matrix)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
            
        Er = self.penalfactor * torch.einsum('geIij,geIij,ge->', Ugrad2, Ugrad2, self.gaussian_weight)


        return Ea + Er
    
    def _get_EpdUe_EpdUe2(self, U, if_onlyforce = False):
        result0 = super()._get_EpdUe_EpdUe2(U, if_onlyforce)

        Ugrad2 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad2 += torch.einsum('geij,eI->geIij',
                                         self.shape_function_d2_gaussian[..., i],
                                         U[self._elems[:, i]])
            
        EmdUgrad2 = 2 * self.penalfactor * Ugrad2

        EmdUe = torch.einsum('geIij,geija->aIe', EmdUgrad2,
                                self._dN2W)
        
        if if_onlyforce:
            return EmdUe + result0

        return EmdUe + result0[0], self._EmdUe_2 + result0[1]

class SIMPElementC3D10(torchfea.elements.C3D10, SIMPElement):
    pass

def get_fe_C3D4Less():
    fem = torchfea.FEA_INP()
    # fem.Read_INP(
    #     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
    # )

    # fem.Read_INP(
    #     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
    # )
    import pathlib
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))

    path0 = pathlib.Path(current_path).parent / 'models' / 'C3D4Less.inp'

    fem.read_inp(path0)

    fe = torchfea.from_inp(fem)
    fe.solver = torchfea.solver.StaticImplicitSolver()
    # elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
    # fe.elems['element-0'].set_materials(elems)

    # torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

    fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                    name='pressure_1')

    bc_name = fe.assembly.add_boundary(
        torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

    rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

    fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))
    
    fe.assembly.get_part('final_model').convert_linear_to_quadratic_elements(['C3D4'], ['C3D4'])

    return fe

if __name__ == '__main__':
    fe = torchfea.load_model('tests/test_simp_material/simpmodel.npz')

    # fe = get_fe_C3D4Less()

    fe.assembly.get_load('pressure_1').pressure = 0.06

    fe.initialize()

    result = torchfea.solver.StaticResult.load('tests/test_simp_material/result.npz')

    # fe.assembly.show_all(result.GC)




    part = fe.assembly.get_part('final_model')

    elem_c3d4 = part.elems['C3D4']
    elem_c3d4_simp = SIMPElementC3D10(elem_c3d4._elems_index, elem_c3d4._elems, penalfactor=torch.tensor(1e1))
    elem_c3d4_simp.materials = elem_c3d4.materials

    part.elems['C3D4'] = elem_c3d4_simp


    resultsimp = fe.solve()

    fe.assembly.show_all(resultsimp.GC)



    raise False