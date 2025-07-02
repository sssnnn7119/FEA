import numpy as np
import torch
from .C3base import Element_3D


class C3D8(Element_3D):
    """
    C3D8 - 8-node linear brick, full integration
    
    Local coordinates:
        origin: corner node 0
        g, h, r: local coordinates aligned with element edges starting from node 0
        All coordinates vary from -1 to 1

    Node numbering follows Abaqus convention:
        Bottom face (r=-1):
            0: (-1, -1, -1) - corner
            1: ( 1, -1, -1) - corner
            2: ( 1,  1, -1) - corner
            3: (-1,  1, -1) - corner
        Top face (r=1):
            4: (-1, -1,  1) - corner
            5: ( 1, -1,  1) - corner
            6: ( 1,  1,  1) - corner
            7: (-1,  1,  1) - corner
            
    Face definitions:
        face0: 0123 (Bottom face, r=-1)
        face1: 4567 (Top face, r=1)
        face2: 0154 (Left face, g=-1)
        face3: 1265 (Right face, g=1)
        face4: 0431 (Front face, h=-1)
        face5: 5726 (Back face, h=1)

    Shape functions:
        N_i = 1/8 * (1 + g*g_i) * (1 + h*h_i) * (1 + r*r_i)
        where (g_i, h_i, r_i) are the coordinates of the i-th node
    """

    def __init__(self,
                 elems: torch.Tensor = None,
                 elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)
        self.order = 1

    def initialize(self, fea):
        # Shape function coefficients
        # Linear shape functions for 8-node brick element with coordinates (g,h,r)
        # According to the provided function ordering:
        # 0: constant, 1: g, 2: h, 3: r, 4: g*h, 5: h*r, 6: r*g, ..., 16: g*h*r
        shape_funcs = torch.zeros((8, 20))

        # Node 0: (-1, -1, -1)
        shape_funcs[0, 0] = 1.0  # constant
        shape_funcs[0, 1] = -1.0  # g
        shape_funcs[0, 2] = -1.0  # h
        shape_funcs[0, 3] = -1.0  # r
        shape_funcs[0, 4] = 1.0  # g*h
        shape_funcs[0, 5] = 1.0  # h*r
        shape_funcs[0, 6] = 1.0  # r*g
        shape_funcs[0, 16] = -1.0  # g*h*r

        # Node 1: (1, -1, -1)
        shape_funcs[1, 0] = 1.0  # constant
        shape_funcs[1, 1] = 1.0  # g
        shape_funcs[1, 2] = -1.0  # h
        shape_funcs[1, 3] = -1.0  # r
        shape_funcs[1, 4] = -1.0  # g*h
        shape_funcs[1, 5] = 1.0  # h*r
        shape_funcs[1, 6] = -1.0  # r*g
        shape_funcs[1, 16] = 1.0  # g*h*r

        # Node 2: (1, 1, -1)
        shape_funcs[2, 0] = 1.0  # constant
        shape_funcs[2, 1] = 1.0  # g
        shape_funcs[2, 2] = 1.0  # h
        shape_funcs[2, 3] = -1.0  # r
        shape_funcs[2, 4] = 1.0  # g*h
        shape_funcs[2, 5] = -1.0  # h*r
        shape_funcs[2, 6] = -1.0  # r*g
        shape_funcs[2, 16] = -1.0  # g*h*r

        # Node 3: (-1, 1, -1)
        shape_funcs[3, 0] = 1.0  # constant
        shape_funcs[3, 1] = -1.0  # g
        shape_funcs[3, 2] = 1.0  # h
        shape_funcs[3, 3] = -1.0  # r
        shape_funcs[3, 4] = -1.0  # g*h
        shape_funcs[3, 5] = -1.0  # h*r
        shape_funcs[3, 6] = 1.0  # r*g
        shape_funcs[3, 16] = 1.0  # g*h*r

        # Node 4: (-1, -1, 1)
        shape_funcs[4, 0] = 1.0  # constant
        shape_funcs[4, 1] = -1.0  # g
        shape_funcs[4, 2] = -1.0  # h
        shape_funcs[4, 3] = 1.0  # r
        shape_funcs[4, 4] = 1.0  # g*h
        shape_funcs[4, 5] = -1.0  # h*r
        shape_funcs[4, 6] = -1.0  # r*g
        shape_funcs[4, 16] = 1.0  # g*h*r

        # Node 5: (1, -1, 1)
        shape_funcs[5, 0] = 1.0  # constant
        shape_funcs[5, 1] = 1.0  # g
        shape_funcs[5, 2] = -1.0  # h
        shape_funcs[5, 3] = 1.0  # r
        shape_funcs[5, 4] = -1.0  # g*h
        shape_funcs[5, 5] = -1.0  # h*r
        shape_funcs[5, 6] = 1.0  # r*g
        shape_funcs[5, 16] = -1.0  # g*h*r

        # Node 6: (1, 1, 1)
        shape_funcs[6, 0] = 1.0  # constant
        shape_funcs[6, 1] = 1.0  # g
        shape_funcs[6, 2] = 1.0  # h
        shape_funcs[6, 3] = 1.0  # r
        shape_funcs[6, 4] = 1.0  # g*h
        shape_funcs[6, 5] = 1.0  # h*r
        shape_funcs[6, 6] = 1.0  # r*g
        shape_funcs[6, 16] = 1.0  # g*h*r

        # Node 7: (-1, 1, 1)
        shape_funcs[7, 0] = 1.0  # constant
        shape_funcs[7, 1] = -1.0  # g
        shape_funcs[7, 2] = 1.0  # h
        shape_funcs[7, 3] = 1.0  # r
        shape_funcs[7, 4] = -1.0  # g*h
        shape_funcs[7, 5] = 1.0  # h*r
        shape_funcs[7, 6] = -1.0  # r*g
        shape_funcs[7, 16] = -1.0  # g*h*r

        # Scale by 1/8 for proper normalization of trilinear shape functions
        shape_funcs = shape_funcs * 0.125

        # Derivatives of shape functions with respect to g, h, r
        # The derivatives are constant for linear shape functions
        derivatives = torch.zeros((3, 8, 20))

        # dN/dg derivatives
        for i in range(3):
            derivatives[i] = self._shape_function_derivative(shape_funcs, i)

        # Store shape functions and derivatives
        self.shape_function = [shape_funcs, derivatives]

        self.num_nodes_per_elem = 8

        # Full integration - 2x2x2 = 8 Gaussian points
        self._num_gaussian = 8

        # Standard weight for Gaussian quadrature (1 for each point)
        self.gaussian_weight = torch.ones(8)

        # Gauss points for 2x2x2 integration
        # Use Gauss-Legendre quadrature with points at ±1/sqrt(3)
        p = 1.0 / np.sqrt(3.0)
        p0 = torch.tensor([
            [-p, -p, -p],  # Point 1
            [p, -p, -p],  # Point 2
            [p, p, -p],  # Point 3
            [-p, p, -p],  # Point 4
            [-p, -p, p],  # Point 5
            [p, -p, p],  # Point 6
            [p, p, p],  # Point 7
            [-p, p, p]  # Point 8
        ])

        # Load the Gaussian points for integration
        self._pre_load_gaussian(p0, nodes=fea.nodes)
        super().initialize(fea)

    def find_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        """
        Find the surface elements for a given surface index and element indices.
        
        Args:
            surface_ind: Surface index (0-5)
            elems_ind: Element indices
            
        Returns:
            torch.Tensor: Surface element node indices
        """
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]

        if index_now.shape[0] == 0:
            return torch.empty([0, 4],
                               dtype=torch.long,
                               device=self._elems.device)

        # Return appropriate face nodes according to Abaqus convention
        if surface_ind == 0:  # Bottom face (r=-1): nodes 0,1,2,3
            return self._elems[index_now][:, [0, 1, 2, 3]]
        elif surface_ind == 1:  # Top face (r=1): nodes 4,5,6,7
            return self._elems[index_now][:, [4, 5, 6, 7]]
        elif surface_ind == 2:  # Left face (g=-1): nodes 0,4,7,3
            return self._elems[index_now][:, [0, 4, 7, 3]]
        elif surface_ind == 3:  # Right face (g=1): nodes 1,2,6,5
            return self._elems[index_now][:, [1, 2, 6, 5]]
        elif surface_ind == 4:  # Front face (h=-1): nodes 0,1,5,4
            return self._elems[index_now][:, [0, 1, 5, 4]]
        elif surface_ind == 5:  # Back face (h=1): nodes 3,2,6,7
            return self._elems[index_now][:, [3, 2, 6, 7]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")


class C3D8R(C3D8):
    """
    C3D8R - 8-node linear brick, reduced integration with hourglass control
    
    Local coordinates:
        origin: corner node 0
        g, h, r: local coordinates aligned with element edges starting from node 0
        All coordinates vary from -1 to 1

    Node numbering follows Abaqus convention:
        Bottom face (r=-1):
            0: (-1, -1, -1) - corner
            1: ( 1, -1, -1) - corner
            2: ( 1,  1, -1) - corner
            3: (-1,  1, -1) - corner
        Top face (r=1):
            4: (-1, -1,  1) - corner
            5: ( 1, -1,  1) - corner
            6: ( 1,  1,  1) - corner
            7: (-1,  1,  1) - corner
            
    Face definitions:
        face0: 0123 (Bottom face, r=-1)
        face1: 4567 (Top face, r=1)
        face2: 0154 (Left face, g=-1)
        face3: 1265 (Right face, g=1)
        face4: 0431 (Front face, h=-1)
        face5: 5726 (Back face, h=1)

    Shape functions:
        N_i = 1/8 * (1 + g*g_i) * (1 + h*h_i) * (1 + r*r_i)
        where (g_i, h_i, r_i) are the coordinates of the i-th node
    """

    def __init__(self,
                 elems: torch.Tensor = None,
                 elems_index: torch.Tensor = None):
        super().__init__(elems=elems, elems_index=elems_index)
        self._hg_alpha = 1.0  # Hourglass stabilization parameter

    def initialize(self, fea):
        # Shape function coefficients
        # Linear shape functions for 8-node brick element with coordinates (g,h,r)
        # According to the provided function ordering:
        # 0: constant, 1: g, 2: h, 3: r, 4: g*h, 5: h*r, 6: r*g, ..., 16: g*h*r
        shape_funcs = torch.zeros((8, 20))

        # Node 0: (-1, -1, -1)
        shape_funcs[0, 0] = 1.0  # constant
        shape_funcs[0, 1] = -1.0  # g
        shape_funcs[0, 2] = -1.0  # h
        shape_funcs[0, 3] = -1.0  # r
        shape_funcs[0, 4] = 1.0  # g*h
        shape_funcs[0, 5] = 1.0  # h*r
        shape_funcs[0, 6] = 1.0  # r*g
        shape_funcs[0, 16] = -1.0  # g*h*r

        # Node 1: (1, -1, -1)
        shape_funcs[1, 0] = 1.0  # constant
        shape_funcs[1, 1] = 1.0  # g
        shape_funcs[1, 2] = -1.0  # h
        shape_funcs[1, 3] = -1.0  # r
        shape_funcs[1, 4] = -1.0  # g*h
        shape_funcs[1, 5] = 1.0  # h*r
        shape_funcs[1, 6] = -1.0  # r*g
        shape_funcs[1, 16] = 1.0  # g*h*r

        # Node 2: (1, 1, -1)
        shape_funcs[2, 0] = 1.0  # constant
        shape_funcs[2, 1] = 1.0  # g
        shape_funcs[2, 2] = 1.0  # h
        shape_funcs[2, 3] = -1.0  # r
        shape_funcs[2, 4] = 1.0  # g*h
        shape_funcs[2, 5] = -1.0  # h*r
        shape_funcs[2, 6] = -1.0  # r*g
        shape_funcs[2, 16] = -1.0  # g*h*r

        # Node 3: (-1, 1, -1)
        shape_funcs[3, 0] = 1.0  # constant
        shape_funcs[3, 1] = -1.0  # g
        shape_funcs[3, 2] = 1.0  # h
        shape_funcs[3, 3] = -1.0  # r
        shape_funcs[3, 4] = -1.0  # g*h
        shape_funcs[3, 5] = -1.0  # h*r
        shape_funcs[3, 6] = 1.0  # r*g
        shape_funcs[3, 16] = 1.0  # g*h*r

        # Node 4: (-1, -1, 1)
        shape_funcs[4, 0] = 1.0  # constant
        shape_funcs[4, 1] = -1.0  # g
        shape_funcs[4, 2] = -1.0  # h
        shape_funcs[4, 3] = 1.0  # r
        shape_funcs[4, 4] = 1.0  # g*h
        shape_funcs[4, 5] = -1.0  # h*r
        shape_funcs[4, 6] = -1.0  # r*g
        shape_funcs[4, 16] = 1.0  # g*h*r

        # Node 5: (1, -1, 1)
        shape_funcs[5, 0] = 1.0  # constant
        shape_funcs[5, 1] = 1.0  # g
        shape_funcs[5, 2] = -1.0  # h
        shape_funcs[5, 3] = 1.0  # r
        shape_funcs[5, 4] = -1.0  # g*h
        shape_funcs[5, 5] = -1.0  # h*r
        shape_funcs[5, 6] = 1.0  # r*g
        shape_funcs[5, 16] = -1.0  # g*h*r

        # Node 6: (1, 1, 1)
        shape_funcs[6, 0] = 1.0  # constant
        shape_funcs[6, 1] = 1.0  # g
        shape_funcs[6, 2] = 1.0  # h
        shape_funcs[6, 3] = 1.0  # r
        shape_funcs[6, 4] = 1.0  # g*h
        shape_funcs[6, 5] = 1.0  # h*r
        shape_funcs[6, 6] = 1.0  # r*g
        shape_funcs[6, 16] = 1.0  # g*h*r

        # Node 7: (-1, 1, 1)
        shape_funcs[7, 0] = 1.0  # constant
        shape_funcs[7, 1] = -1.0  # g
        shape_funcs[7, 2] = 1.0  # h
        shape_funcs[7, 3] = 1.0  # r
        shape_funcs[7, 4] = -1.0  # g*h
        shape_funcs[7, 5] = 1.0  # h*r
        shape_funcs[7, 6] = -1.0  # r*g
        shape_funcs[7, 16] = -1.0  # g*h*r

        # Scale by 1/8 for proper normalization of trilinear shape functions
        shape_funcs = shape_funcs * 0.125

        # Derivatives of shape functions with respect to g, h, r
        # The derivatives are constant for linear shape functions
        derivatives = torch.zeros((3, 8, 20))

        # dN/dg derivatives
        for i in range(3):
            derivatives[i] = self._shape_function_derivative(shape_funcs, i)

        # Store shape functions and derivatives
        self.shape_function = [shape_funcs, derivatives]

        self.num_nodes_per_elem = 8
        self._num_gaussian = 1  # Reduced integration - single point at element center
        self.gaussian_weight = torch.tensor(
            [8.0])  # Weight for single Gauss point (volume = 8)
        # Gauss point at center of element (0, 0, 0) in local coordinates
        p0 = torch.tensor([[0.0, 0.0, 0.0]])

        # Initialize hourglass control parameters
        self._initialize_hourglass_control()

        # Load the Gaussian points for integration
        self._pre_load_gaussian(p0, nodes=fea.nodes)
        super().initialize(fea)

    def _initialize_hourglass_control(self):
        """
        Initialize hourglass control for C3D8R element.
        
        Defines the hourglass modes and parameters based on Flanagan-Belytschko algorithm.
        """
        # Define hourglass modes (4 modes for 3D element)
        # Hourglass modes represent deformation patterns that aren't captured by reduced integration
        self._hg_modes = torch.tensor([
            [1., -1., 1., -1., -1., 1., -1., 1.],  # Mode 1: g-hourglass mode
            [1., 1., -1., -1., -1., -1., 1., 1.],  # Mode 2: h-hourglass mode
            [1., -1., -1., 1., -1., 1., 1., -1.],  # Mode 3: r-hourglass mode
            [1., 1., 1., 1., -1., -1., -1., -1.]  # Mode 4: ghr-hourglass mode
        ])

        # Define properties needed for hourglass control
        self._num_hg_modes = 4

        # Initialize tensors for hourglass calculations
        # Will be populated in structural_Force and potential_Energy methods
        self._hg_energy = None
        self._hg_forces = None
        self._hg_stiffness = None

    def _calculate_hourglass_parameters(self, U):
        """
        Calculate parameters for hourglass control.
        
        Args:
            U: Displacement field [N, 3]
            
        Returns:
            tuple: (hg_gamma, B0, shear_modulus, element_volume)
        """
        # Get nodal coordinates and displacements for each element
        disp = torch.zeros([self.num_nodes_per_elem, self._elems.shape[0], 3])

        for i in range(self.num_nodes_per_elem):
            disp[i] = U[self._elems[:, i]]

        # Element volume (from Gaussian weights)
        element_volume = self.gaussian_weight.sum(0)

        # Calculate physical shape function derivatives at the integration point

        # Get material properties (assuming isotropic material for simplicity)

        # Extract an approximate shear modulus from the material stiffness tensor
        shear_modulus = self.materials.mu.flatten()

        # Calculate hourglass parameters for each mode
        # γₐᵢ = ∑ᵦ Γₐᵦ uᵦᵢ (hourglass mode projection coefficients)
        hg_gamma = torch.zeros([self._elems.shape[0], self._num_hg_modes, 3])

        for a in range(self._num_hg_modes):
            for i in range(3):  # x, y, z components
                # For each hourglass mode, calculate its contribution based on displacement field
                hg_gamma[:, a, i] = torch.einsum('g,ge->e', self._hg_modes[a],
                                                 disp[:, :, i])

        return hg_gamma, shear_modulus, element_volume

    def _calculate_hourglass_energy(self, U):
        """
        Calculate the hourglass energy for stabilization.
        
        Args:
            U: Displacement field [N, 3]
            
        Returns:
            torch.Tensor: Hourglass energy
        """
        # Get parameters
        hg_gamma, shear_modulus, element_volume = self._calculate_hourglass_parameters(
            U)

        # Calculate hourglass energy
        # W_hg = α * G * V * ∑ₐᵢ γₐᵢ²
        # Where α is a scaling factor, G is shear modulus, V is element volume
        hg_energy = torch.einsum('eai,eai,e,e->e', hg_gamma, hg_gamma,
                                 shear_modulus, element_volume)

        # Scale by hourglass parameter (typically between 0.01 and 0.10)
        hg_energy = 0.5 * self._hg_alpha * hg_energy

        return hg_energy.sum()

    def _calculate_hourglass_forces(self, U):
        """
        Calculate hourglass forces for stabilization.
        
        Args:
            U: Displacement field [N, 3]
            
        Returns:
            torch.Tensor: Hourglass forces [n_nodes*3]
        """
        # Get parameters
        hg_gamma, shear_modulus, element_volume = self._calculate_hourglass_parameters(
            U)

        # Calculate hourglass forces
        # Fᵦⱼ = α * G * V * ∑ₐᵢ γₐᵢ * Γₐᵦ * δᵢⱼ
        hg_forces = torch.zeros(
            [self._elems.shape[0], self.num_nodes_per_elem, 3])

        for b in range(self.num_nodes_per_elem):
            for j in range(3):  # x, y, z components
                for a in range(self._num_hg_modes):
                    hg_forces[:, b,
                              j] += hg_gamma[:, a,
                                             j] * shear_modulus * element_volume * self._hg_modes[
                                                 a, b]

        # Scale by hourglass parameter
        hg_forces = self._hg_alpha * hg_forces

        # Return reshaped forces for assembly
        return hg_forces

    def _calculate_hourglass_stiffness(self, U):
        """
        Calculate hourglass stiffness for stabilization.
        
        Args:
            U: Displacement field [N, 3]
            
        Returns:
            torch.Tensor: Hourglass stiffness contribution
        """
        # Get parameters
        hg_gamma, shear_modulus, element_volume = self._calculate_hourglass_parameters(
            U)

        # Calculate hourglass stiffness
        # Kᵦⱼᵧₖ = α * G * V * ∑ₐ Γₐᵦ * Γₐᵧ * δⱼₖ
        hg_stiffness = torch.zeros([
            self._elems.shape[0], self.num_nodes_per_elem, 3,
            self.num_nodes_per_elem, 3
        ])

        for b in range(self.num_nodes_per_elem):
            for g in range(self.num_nodes_per_elem):
                for j in range(3):  # x, y, z components
                    for k in range(3):  # x, y, z components
                        if j == k:  # Only diagonal terms (δⱼₖ)
                            for a in range(self._num_hg_modes):
                                hg_stiffness[:, b, j, g, k] += torch.einsum('e,e->e',
                                                                         shear_modulus,
                                                                         element_volume) * \
                                                            self._hg_modes[a, b] * self._hg_modes[a, g]

        # Scale by hourglass parameter
        hg_stiffness = self._hg_alpha * hg_stiffness

        # Reshape for assembly (match the shape expected by the parent's structural_Force method)
        hg_stiffness_flat = hg_stiffness.reshape(self._elems.shape[0],
                                                 self.num_nodes_per_elem * 3,
                                                 self.num_nodes_per_elem * 3)

        return hg_stiffness_flat

    def potential_Energy(self, RGC: list[torch.Tensor]):
        """
        Calculate potential energy with hourglass stabilization.
        
        Args:
            RGC: List of tensors with displacement fields
            
        Returns:
            torch.Tensor: Total potential energy
        """
        # Get standard potential energy from parent class
        Ea = super().potential_Energy(RGC)

        # Add hourglass energy
        U = RGC[0].reshape([-1, 3])
        Ehg = self._calculate_hourglass_energy(U)

        # Total energy is sum of standard energy and hourglass energy
        return Ea + Ehg

    def structural_Force(self, RGC: list[torch.Tensor]):
        """
        Calculate structural forces with hourglass stabilization.
        
        Args:
            RGC: List of tensors with displacement fields
            
        Returns:
            tuple: (force_indices, residual_force, stiffness_indices, stiffness_values)
        """
        # Get standard forces and stiffness from parent class
        indices_force, Relement, indices_matrix, values = super(
        ).structural_Force(RGC)

        # Add hourglass forces and stiffness
        U = RGC[0].reshape([-1, 3])

        # Calculate hourglass forces
        hg_forces = self._calculate_hourglass_forces(U)

        # Reshape and assemble hourglass forces
        hg_forces_flat = hg_forces.reshape(-1, self.num_nodes_per_elem * 3)
        Rhg_element = hg_forces_flat.flatten()

        # Add hourglass forces to residual
        Relement = Relement + Rhg_element

        # Calculate hourglass stiffness
        hg_stiffness_flat = self._calculate_hourglass_stiffness(U)

        # Reshape hourglass stiffness for assembly
        Khg_element = hg_stiffness_flat.flatten()

        # Add hourglass stiffness to values
        values = values + torch.zeros(
            [self._indices_matrix.shape[1]]).scatter_add(
                0, self._index_matrix_coalesce, Khg_element)

        return indices_force, Relement, indices_matrix, values
