import torch

from .base import Materials_Base


class LinearElastic(Materials_Base):
    """
    Linear elastic material model adapted for large deformation.
    
    This model uses Young's modulus (E) and Poisson's ratio (nu) as inputs
    and implements a hyperelastic formulation based on linear elasticity that
    can handle large deformations.
    """

    def __init__(self, E: torch.Tensor | float,
                 nu: torch.Tensor | float) -> None:
        """
        Initialize a linear elastic material for large deformation.
        
        Args:
            E: Young's modulus
            nu: Poisson's ratio
        """
        super().__init__()

        self.type = 2  # Material type 2 for linear elasticity

        # Convert scalar inputs to tensors if needed
        if isinstance(E, float):
            E = torch.tensor([E], dtype=torch.float32)

        if isinstance(nu, float):
            nu = torch.tensor([nu], dtype=torch.float32)

        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        
        # Pre-compute Lamé parameters
        self.lambda_ = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))  # Shear modulus (second Lamé parameter)

    def _broadcast_param(self, x: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """Broadcast material parameter to [g, e] shape."""
        g, e = F.shape[0], F.shape[1]
        if x.dim() == 0 or x.numel() == 1:
            return x.reshape(1, 1).expand(g, e)
        if x.dim() == 1:
            if x.shape[0] == g:
                return x.view(g, 1).expand(g, e)
            if x.shape[0] == e:
                return x.view(1, e).expand(g, e)
            raise ValueError(f"Cannot broadcast parameter with shape {tuple(x.shape)} to [{g}, {e}]")
        if x.dim() == 2 and x.shape[0] == g and x.shape[1] == e:
            return x
        raise ValueError(f"Unsupported parameter shape {tuple(x.shape)}")

    def strain_energy_density_C3(self,
                                 F: torch.Tensor = None,
                                 I1: torch.Tensor = None,
                                 J: torch.Tensor = None):
        """
        Compute the strain energy density for large deformation linear elasticity.
        
        For large deformations, we use the Saint Venant-Kirchhoff model:
        W = (lambda/2)(tr(E))^2 + mu*tr(E^2)
        where E = 1/2*(F^T*F - I) is the Green-Lagrange strain tensor
        
        Args:
            F: Deformation gradient
            I1: First invariant (optional)
            J: Jacobian determinant (optional)
            
        Returns:
            Strain energy density
        """
        batch_size, elem_size = F.shape[0], F.shape[1]

        # Green-Lagrange strain tensor E = 1/2*(C - I)
        C = torch.einsum('geij,gejk->geik', F, F)
        I_tensor = torch.eye(3, device=F.device, dtype=F.dtype).reshape(1, 1, 3, 3)
        E = 0.5 * (C - I_tensor)

        # Trace of E: tr(E)
        tr_E = torch.diagonal(E, dim1=-2, dim2=-1).sum(-1)

        # Compute E^2
        E_squared = torch.einsum('geij,gejk->geik', E, E)

        # Trace of E^2: tr(E^2)
        tr_E_squared = torch.diagonal(E_squared, dim1=-2, dim2=-1).sum(-1)

        # Broadcast Lamé parameters
        lambda_ = self._broadcast_param(self.lambda_.to(F.device, F.dtype), F)
        mu = self._broadcast_param(self.mu.to(F.device, F.dtype), F)

        # W = (lambda/2)(tr(E))^2 + mu*tr(E^2)
        W = 0.5 * lambda_ * tr_E**2 + mu * tr_E_squared

        return W

    def material_Constitutive_C3(self, F: torch.Tensor):
        """
        S-F description interface.

        Args:
            F: deformation gradient, shape [g, e, 3, 3]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - S: 2nd Piola stress, shape [g, e, 3, 3]
                - C_ref: dS/dE in reference configuration, shape [g, e, 3, 3, 3, 3]
        """
        batch_size, elem_size = F.shape[0], F.shape[1]

        # Right Cauchy-Green deformation tensor
        C = torch.einsum('geij,gejk->geik', F, F)

        # Identity tensor
        I_tensor = torch.eye(3, device=F.device, dtype=F.dtype).reshape(1, 1, 3, 3)

        # Green-Lagrange strain tensor
        E = 0.5 * (C - I_tensor)

        # Trace of Green-Lagrange strain tensor
        tr_E = E.diagonal(dim1=-2, dim2=-1).sum(-1)

        # Broadcast Lamé parameters
        lambda_ = self._broadcast_param(self.lambda_.to(F.device, F.dtype), F)
        mu = self._broadcast_param(self.mu.to(F.device, F.dtype), F)

        lambda4 = lambda_.view(batch_size, elem_size, 1, 1)
        mu4 = mu.view(batch_size, elem_size, 1, 1)

        # 2nd Piola stress S = lambda tr(E) I + 2 mu E
        S = lambda4 * tr_E.view(batch_size, elem_size, 1, 1) * I_tensor + 2.0 * mu4 * E

        # Material elasticity in reference configuration:
        # C_ref_{IJKL} = lambda δIJ δKL + mu(δIK δJL + δIL δJK)
        C_ref = torch.zeros([batch_size, elem_size, 3, 3, 3, 3], device=F.device, dtype=F.dtype)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        delta_ij = 1.0 if i == j else 0.0
                        delta_kl = 1.0 if k == l else 0.0
                        delta_ik = 1.0 if i == k else 0.0
                        delta_jl = 1.0 if j == l else 0.0
                        delta_il = 1.0 if i == l else 0.0
                        delta_jk = 1.0 if j == k else 0.0

                        C_ref[..., i, j, k, l] = (
                            lambda_ * delta_ij * delta_kl
                            + mu * (delta_ik * delta_jl + delta_il * delta_jk)
                        )

        return S, C_ref