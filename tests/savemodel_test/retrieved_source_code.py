#========= Source code for Serializable =========#
class Serializable():

    _serialized_attributes: list[str] = []
    """List of attribute names to be serialized."""

    _subclasses: dict[str, 'Serializable'] = {}
    """Registry of subclasses for factory method."""

    _subclass_source_code: dict[str, str] = {}
    """Registry of subclass source code for debugging and reproducibility."""

    def __init_subclass__(cls):
        """Register subclasses in the class registry for factory method."""
        cls._subclasses[cls._get_obj_name()] = cls
        cls._subclass_source_code[cls._get_obj_name()] = cls._get_source_code()

    @classmethod
    def _get_source_code(cls) -> str:
        """Get source code of this class and all its ancestor classes.

        Walks the MRO (Method Resolution Order) and concatenates the source
        code of each class from the root base down to this class.
        Uses ``inspect.getsource``, so the source file must be available.

        Returns:
            str: Concatenated source code of all classes in the MRO.
        """
        source_parts: list[str] = []
        # seen: set[str] = set()
        # for klass in reversed(cls.__mro__):
        #     if klass is object or klass is Serializable:
        #         continue
        #     if klass.__name__ in seen:
        #         continue
        #     seen.add(klass.__name__)
        #     source_parts.append(inspect.getsource(klass))
        source_parts.append(inspect.getsource(cls))
        return "\n".join(source_parts)

    def __init__(self) -> None:
        super().__init__()

    @property
    def serialized_attributes(self):
        """Get the list of attributes to be serialized."""
        serialized_attrs = []
        if not self._serialized_attributes:
            serialized_attrs = [attr for attr in self.__dict__.keys() if not attr.startswith('__')]
        else:
            serialized_attrs = self._serialized_attributes
        return serialized_attrs
    
    @classmethod
    def _get_obj_name(cls):
        """Get the name of the object's class, including the mro."""

        mro = inspect.getmro(cls)

        name: list[str] = []
        for klass in mro:
            if klass is object:
                continue
            name.append(klass.__name__)
        name = ".".join(reversed(name))
        return name

    @staticmethod
    def _serialize_obj(obj):
        """Helper function to serialize an object."""
        if isinstance(obj, torch.Tensor):
            return (obj.detach().cpu().numpy(), type(obj).__name__)
        elif issubclass(type(obj), Serializable):
            return obj._serialize()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return (obj, type(obj).__name__)
        elif isinstance(obj, (list, tuple)):
            return ([Serializable._serialize_obj(item) for item in obj], type(obj).__name__)
        elif isinstance(obj, dict):
            return ({key: Serializable._serialize_obj(value) for key, value in obj.items()}, type(obj).__name__)
        elif isinstance(obj, np.ndarray):
            return (obj, type(obj).__name__)
        else:
            return ()

    def _serialize(self) -> dict:
        """
        Serialize the object to a dictionary.

        Returns:
            dict: A dictionary containing the serialized attributes.
        """


        serialized_data = {}

        all_attributes = [attr for attr in self.__dict__.keys() if not attr.startswith('__')]
        selected_attributes = self.serialized_attributes
        
        selected_attributes = list(set(selected_attributes))

        for attr in all_attributes:
            if attr in selected_attributes:
                value = getattr(self, attr)
                sub_serialized = self._serialize_obj(value)
                if sub_serialized:
                    serialized_data[attr] = sub_serialized
            else:
                serialized_data[attr] = (None, 'NoneType')


        return (serialized_data, self._get_obj_name())
    
    @staticmethod
    def _deserialize_obj(data: tuple):
        """Helper function to deserialize an object."""
        if not data:
            return None
        value, type_name = data
        if type_name in Serializable._subclasses:
            return Serializable._subclasses[type_name]._deserialize((value, type_name))
        elif type_name == 'Tensor':
            data_now = torch.from_numpy(value).to(torch.get_default_device())
            if data_now.dtype == torch.float64 or data_now.dtype == torch.float32:
                data_now = data_now.to(torch.get_default_dtype())
            return data_now
        elif type_name in ['int', 'float', 'str', 'bool', 'NoneType']:
            return value
        elif type_name in ['list', 'tuple']:
            return [Serializable._deserialize_obj(item) for item in value]
        elif type_name == 'dict':
            return {key: Serializable._deserialize_obj(val) for key, val in value.items()}
        elif type_name == 'ndarray':
            return value
        elif type_name == 'NoneType':
            return None
        else:
            raise ValueError(f"Unknown type name '{type_name}' during deserialization.")
    
    @classmethod
    def _deserialize(cls, data: tuple[dict, str]):
        """
        Deserialize the object from a dictionary.

        Args:
            data (dict): A dictionary containing the serialized attributes.
        """
        serialized_data, class_name = data
        if class_name not in cls._subclasses:
            raise ValueError(f"Unknown class name '{class_name}' during deserialization.")
        
        obj = cls._subclasses[class_name].__new__(cls._subclasses[class_name])
        for attr, value in serialized_data.items():
            deserialized_value = cls._deserialize_obj(value)
            setattr(obj, attr, deserialized_value)
        return obj


#========= Source code for Serializable.Materials_Base =========#
class Materials_Base(Serializable):

    def __init__(self) -> None:
        super().__init__()
        pass

    def material_Constitutive_C3(self, F, J, Jneg, invF, I1):
        pass

    def strain_energy_density_C3(self, F):
        pass


#========= Source code for Serializable.Materials_Base.NeoHookean =========#
class NeoHookean(Materials_Base):

    _serialized_attributes: list[str] = ['_mu', '_kappa']

    def __init__(self, mu: torch.Tensor | float,
                 kappa: torch.Tensor | float) -> None:

        super().__init__()

        self.type = 1

        # if mu is scalar, then flag it as a constant
        if type(mu) == float:
            mu = torch.tensor([mu], dtype=torch.float32)

        if type(kappa) == float:
            kappa = torch.tensor([kappa], dtype=torch.float32)

        self._mu = mu
        self._kappa = kappa

    @property
    def mu(self) -> torch.Tensor:
        return self._mu
    
    @mu.setter
    def mu(self, value: torch.Tensor | float) -> None:
        if type(value) == float:
            value = torch.tensor([value], dtype=torch.float32)
        self._mu = value

    @property
    def kappa(self) -> torch.Tensor:
        return self._kappa
    
    @kappa.setter
    def kappa(self, value: torch.Tensor | float) -> None:
        if type(value) == float:
            value = torch.tensor([value], dtype=torch.float32)
        self._kappa = value

    def strain_energy_density_C3(self,
                                 F: torch.Tensor = None):

        
        J = F.det()
        I1 = (F**2).sum([-1, -2]) * J**(-2 / 3)
        W = self.mu    / 2 * (I1 - 3) + \
            self.kappa / 2 * (J  - 1)**2
        return W

    def material_Constitutive_C3(self,
                                 F: torch.Tensor,
                                 J: torch.Tensor = None,
                                 Jneg: torch.Tensor = None,
                                 invF: torch.Tensor = None,
                                 I1: torch.Tensor = None):

        if J is None:
            invF = F.inverse()
            J = F.det()
            Jneg = J**(-2 / 3)
            I1 = (F**2).sum([-1, -2]) * Jneg

        J = J.view(J.shape[0], J.shape[1], 1, 1)
        Jneg = Jneg.view(J.shape[0], J.shape[1], 1, 1)
        I1 = I1.view(J.shape[0], J.shape[1], 1, 1)

        if self.mu.dim() == 0 or self.mu.numel() == 1:
            mu = self.mu.view(1, 1, 1, 1)
            kappa = self.kappa.view(1, 1, 1, 1)
        else:
            mu = self.mu.view(self.mu.shape[0], self.mu.shape[1], 1, 1)
            kappa = self.kappa.view(self.kappa.shape[0], self.kappa.shape[1], 1, 1)

        muJneg = mu * Jneg
        FtMuJneg = F.transpose(-1, -2) * muJneg
        muI1invF = mu * I1 * invF
        kappaJinvF = kappa * J * invF

        s = torch.zeros_like(F)
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3])

        s = FtMuJneg + (-1 / 3 * muI1invF + kappaJinvF * (J - 1))

        C = torch.einsum(
            'geij,gelk->geijkl',
            -2 / 3 * FtMuJneg + kappaJinvF * (2 * J - 1) + 2 / 9 * muI1invF,
            invF)

        C += torch.einsum('geij,gekl->geijkl',
                                         -2 / 3 * muJneg * invF, F)
        C += torch.einsum('geik,gelj->geijkl',
                                         (1 / 3 * muI1invF - kappaJinvF *
                                          (J - 1)), invF)

        for m in range(3):
            for n in range(3):
                C[:, :, m, n, n, m] += muJneg[:, :, 0, 0]

        return s, C


#========= Source code for Serializable.Materials_Base.NeoHookean.NeoHookeanLnJ =========#
class NeoHookeanLnJ(NeoHookean):

    def strain_energy_density_C3(self,
                                 F: torch.Tensor = None):

        
        J = F.det()
        I1 = (F**2).sum([-1, -2]) * J**(-2 / 3)
        W = self.mu    / 2 * (I1 - 3) + \
            self.kappa / 2 * (torch.log(J))**2
        return W

    def material_Constitutive_C3(self,
                                 F: torch.Tensor,
                                 J: torch.Tensor = None,
                                 Jneg: torch.Tensor = None,
                                 invF: torch.Tensor = None,
                                 I1: torch.Tensor = None):

        if J is None:
            invF = F.inverse()
            J = F.det()
            Jneg = J**(-2 / 3)
            I1 = (F**2).sum([-1, -2]) * Jneg

        J = J.view(J.shape[0], J.shape[1], 1, 1)
        Jneg = Jneg.view(J.shape[0], J.shape[1], 1, 1)
        I1 = I1.view(J.shape[0], J.shape[1], 1, 1)

        if self.mu.dim() == 0 or self.mu.numel() == 1:
            mu = self.mu.view(1, 1, 1, 1)
            kappa = self.kappa.view(1, 1, 1, 1)
        else:
            mu = self.mu.view(self.mu.shape[0], self.mu.shape[1], 1, 1)
            kappa = self.kappa.view(self.kappa.shape[0], self.kappa.shape[1], 1, 1)

        muJneg = mu * Jneg
        FtMuJneg = F.transpose(-1, -2) * muJneg
        muI1invF = mu * I1 * invF

        s = torch.zeros_like(F)
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3])

        s = FtMuJneg + -1 / 3 * muI1invF + kappa * torch.log(J) * invF

        C = torch.einsum(
            'geij,gelk->geijkl',
            -2 / 3 * FtMuJneg + kappa * invF + 2 / 9 * muI1invF,
            invF)

        C += torch.einsum('geij,gekl->geijkl',
                                         -2 / 3 * muJneg * invF, F)
        C += torch.einsum('geik,gelj->geijkl',
                                         (1 / 3 * muI1invF - kappa * torch.log(J) *
                                          invF), invF)

        for m in range(3):
            for n in range(3):
                C[:, :, m, n, n, m] += muJneg[:, :, 0, 0]

        return s, C


#========= Source code for Serializable.Materials_Base.LinearElastic =========#
class LinearElastic(Materials_Base):
    """
    Linear elastic material model adapted for large deformation.
    
    This model uses Young's modulus (E) and Poisson's ratio (nu) as inputs
    and implements a hyperelastic formulation based on linear elasticity that
    can handle large deformations.
    """

    _serialized_attributes: list[str] = ['E', 'nu']

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


#========= Source code for Serializable.BaseElement =========#
class BaseElement(Serializable):
    
    _serialized_attributes: list[str] = ['_elems_index', '_elems', '_density', 'materials']

    shape_function: list[torch.Tensor]
    """
        the shape functions of the element
    """

    _num_gaussian: int
    """
        the number of guassian points
    """

    num_nodes_per_elem: int
    """ 
        the number of nodes per element
    """

    gaussian_weight_ref: torch.Tensor
    """        the weight of each guassian point"""

    gaussian_coordinates: torch.Tensor
    """the coordinates of gaussian points in the reference space"""


    def __init__(self, elems_index: torch.Tensor, elems: torch.Tensor) -> None:

        super().__init__()
        self.shape_function: list[torch.Tensor]
        """
            the shape of shape_function 
        """
        
        self.shape_function_gaussian: list[torch.Tensor]
        """
            the shape of shape_function at each guassian point
        """
        
        self.gaussian_weight: torch.Tensor
        """
        the weight of each guassian point
            [
                g, the num of guassian point
            ]
        """
        self._elems_index = elems_index
        """
            the index of the element
        """
        self._elems = elems
        """
            [elem, N]\n
            the element connectivity 
        """

        # Materials are managed as a dict so one element can aggregate multiple
        # material contributions in energy/force/stiffness calculations.
        self.materials: dict[str, materials.Materials_Base] = {}

        self._indices_matrix: torch.Tensor
        """
            the coo index of the stiffness matricx of structural stress
        """

        self._indices_force: torch.Tensor
        """
            the coo index of the tructural stress
        """

        self._index_matrix_coalesce: torch.Tensor = None
        """
            the start index of the stiffness matricx of structural stress
        """

        self._density: torch.Tensor = None
        """
            the density of the element
        """

        cls = self.__class__
        cls.shape_function = [cls.shape_function[i].to(torch.get_default_device()).to(torch.get_default_dtype()) for i in range(len(cls.shape_function))]
        cls.gaussian_weight_ref = cls.gaussian_weight_ref.to(torch.get_default_device()).to(torch.get_default_dtype())
        cls.gaussian_coordinates = cls.gaussian_coordinates.to(torch.get_default_device()).to(torch.get_default_dtype())

    @property
    def density(self) -> torch.Tensor:
        return self._density
    
    @density.setter
    def density(self, value: np.ndarray | torch.Tensor):
        if isinstance(value, np.ndarray):
            value = torch.tensor(value)
        self._density = value

    def get_gaussian_points(self, nodes: torch.Tensor) -> torch.Tensor:
        """
            get the gaussian points of the element
        """
        raise NotImplementedError('The gaussian points of the element is not implemented yet')

    def initialize(self, nodes: torch.Tensor, *args, **kwargs):
        pass
    
    def get_mass_matrix(self,rotation_matrix:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            get the mass matrix of the element
        Returns:
            indices (torch.Tensor): the indices of the mass matrix
            M (torch.Tensor): the mass matrix of the element
        """
        raise NotImplementedError('The mass matrix of the element is not implemented yet')

    def potential_Energy(self, RGC: torch.Tensor):
        pass

    def structural_Force(self, RGC: torch.Tensor,rotation_matrix:torch.Tensor, if_onlyforce: bool = False, *args, **kwargs):
        pass

    def _iter_material_values(self):
        """Iterate materials with backward compatibility for legacy single-material assignment."""
        mats = self.materials
        if isinstance(mats, dict):
            return mats.values()

        if isinstance(mats, materials.Materials_Base):
            # Backward compatibility if external code directly assigns single material
            return [mats]

        raise TypeError("materials must be a dict[str, Materials_Base] or a Materials_Base instance")

    def set_materials(self, mat: materials.Materials_Base | dict[str, materials.Materials_Base], name=None):
        """
            Set materials of the element.

            Args:
                mat: one material or a dict of materials
                name: key when setting a single material; auto-generated if None
        """

        if isinstance(mat, dict):
            self.materials = dict(mat)
            return

        if not isinstance(mat, materials.Materials_Base):
            raise TypeError("mat must be Materials_Base or dict[str, Materials_Base]")

        if name is None:
            number = len(self.materials) if isinstance(self.materials, dict) else 0
            name = f"material-{number}"

        if not isinstance(self.materials, dict):
            # Normalize legacy assignment to dict on first set
            self.materials = {}

        self.materials[name] = mat

    def delete_material(self, name: str | None = None):
        """Delete one material by name, or clear all when name is None."""
        mats = self.materials

        if isinstance(mats, dict):
            if name is None:
                mats.clear()
                return
            if name not in mats:
                raise ValueError(f"Material '{name}' not found in this element")
            del mats[name]
            return

        if isinstance(mats, materials.Materials_Base):
            if name is not None:
                raise ValueError("Legacy single-material mode does not support named deletion")
            self.materials = {}
            return

        raise TypeError("materials must be a dict[str, Materials_Base] or a Materials_Base instance")
        
    def set_required_DoFs(
            self, RGC_remain_index: np.ndarray) -> np.ndarray:
        """
        Modify the RGC_remain_index
        """
        
    def extract_surface(self, surface_ind: int, elems_ind: np.ndarray):
        """
        Find the surface of the element

        Args:
            surface_ind (int): the index of the surface
            elems_ind (np.ndarray): the index of the element
        
        Returns:
            list[BaseSurface]: a list of surface elements
        """
        return []
    
    def set_order(self, order: int):
        """
        set the order of the element
        Args:
            order (int): the order of the element
        """
        raise NotImplementedError('The order of the element is not implemented yet')
    
    def refine_RGC(self, RGC: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
            refine the RGC of the element
        """
        return RGC


#========= Source code for Serializable.BaseElement.Element_3D =========#
class Element_3D(BaseElement):



    num_surfaces: int
    """
        the number of surfaces of the element
    """

    def __init_subclass__(cls):
        super().__init_subclass__()

        if hasattr(cls, 'shape_function'):
            cls.shape_function[0] = cls.shape_function[0].to(torch.get_default_device()).to(torch.get_default_dtype())
            cls.shape_function.append(torch.stack([
                    cls._shape_function_derivative(cls.shape_function[0], 0),
                    cls._shape_function_derivative(cls.shape_function[0], 1),
                    cls._shape_function_derivative(cls.shape_function[0], 2),
                ],
                            dim=0).to(torch.get_default_device()).to(torch.get_default_dtype()))
            
            cls.shape_function.append(torch.zeros(
                [3, 3, cls.shape_function[0].shape[0], cls.shape_function[0].shape[1]]))
            for i in range(3):
                for j in range(3):
                    cls.shape_function[2][i, j] = cls._shape_function_derivative(cls.shape_function[1][i], j).to(torch.get_default_device()).to(torch.get_default_dtype())

    def __init__(self, elems_index: torch.Tensor,
                 elems: torch.Tensor) -> None:
        super().__init__(elems_index, elems)

        self.shape_function_d2_gaussian: torch.Tensor
        """
            the second derivative of the shape function of each guassian point
                [
                    g: guassian point
                    e: element
                    i: derivative index 0
                    j: derivative index 1
                    a: a-th node
                ]
        """

        self.shape_function_d1_gaussian: torch.Tensor
        """
            the shape functions of each guassian point
                [
                    g: guassian point
                    e: element
                    i: derivative
                    a: a-th node
                ]
        """

        self.shape_function_d0_gaussian: torch.Tensor
        """the shape functions of each guassian point [guassian, element, node]"""

        self._dNW: torch.Tensor
        """the derivative of the shape function multiplied by the guassian weight [guassian, element, derivative, node]"""

        self._dNdNW: torch.Tensor
        """the derivative of the shape function multiplied by the guassian weight [guassian, element, derivative, node, derivative, node]"""


    def initialize(self, nodes: torch.Tensor, *args, **kwargs) -> None:

        super().initialize(nodes, *args, **kwargs)

        # pre load the gaussian points and its weight for the element, which will be used in the FEA calculation
        self._pre_load_gaussian(nodes=nodes)

        # coo index of the stiffness matricx of structural stress

        index0_ = torch.stack([
                self._elems.T.reshape([self.num_nodes_per_elem, 1, 1, 1, -1]).repeat([1, 3, self.num_nodes_per_elem, 3, 1]),
                torch.arange(3).reshape([1, 3, 1, 1, 1]).repeat([self.num_nodes_per_elem, 1, self.num_nodes_per_elem, 3, self._elems.shape[0]]),
                self._elems.T.reshape([1, 1, self.num_nodes_per_elem, 1, -1]).repeat([self.num_nodes_per_elem, 3, 1, 3, 1]),
                torch.arange(3).reshape([1, 1, 1, 3, 1]).repeat([self.num_nodes_per_elem, 3, self.num_nodes_per_elem, 1, self._elems.shape[0]])
            ], dim=0).reshape([4, -1])
        index0 = torch.zeros([2, index0_.shape[1]], dtype=torch.int64)
        index0[0] = index0_[0] * 3 + index0_[1]
        index0[1] = index0_[2] * 3 + index0_[3]

        # some trick to get the unique index and accelerate the calculation
        scaler = index0.max() + 1
        index1 = index0[0] * scaler + index0[1]
        index_sorted_matrix = index1.argsort()
        index2 = index1[index_sorted_matrix]
        index_unique, self._index_matrix_coalesce = torch.unique_consecutive(
            index2, return_inverse=True)

        inverse_index = torch.zeros_like(index_sorted_matrix,
                                         dtype=torch.int64)
        inverse_index[index_sorted_matrix] = torch.arange(
            0, index_sorted_matrix.max() + 1, dtype=torch.int64)

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

    def _pre_load_gaussian(self, nodes: torch.Tensor):
        """
        Pre-compute shape function values & derivatives at Gaussian points.
        Uses isoparametric mapping: ξ (reference) → x (physical).

        Key steps:
          1. N_a(ξ)   : shape function values at Gauss points
          2. ∂N_a/∂ξ  : shape function gradients w.r.t. reference coords
          3. J = ∂x/∂ξ : Jacobian of isoparametric mapping
          4. ∂N_a/∂x = J^{-1} · ∂N_a/∂ξ : push-forward to physical gradients
          5. dV = det(J) · dξ : integration weight in physical space

        Args:
            nodes: [p, 3], global nodal coordinates x_i^a in physical space
        """

        # —— Step 1: polynomial basis p(ξ) at Gaussian points ——
        # pp[g, m]: m-th monomial term evaluated at Gauss point ξ_g
        pp = self._get_interpolation_coordinates(self.gaussian_coordinates)

        # ∂N_a/∂x_i [g, e, i, a],  N_a [g, e, a],  det(J) [g, e]
        det_Jacobian = torch.zeros([self._num_gaussian, self._elems.shape[0]])

        elem_now = self._elems

        # —— Step 2: Jacobian J_{ij} = ∂x_i/∂ξ_j ——
        # isoparametric mapping: x_i(ξ) = Σ_a N_a(ξ) · x_i^a
        # => J_{ij} = Σ_a (∂N_a/∂ξ_j) · x_i^a
        Jacobian = torch.zeros([self._num_gaussian, elem_now.shape[0], 3, 3])
        shape1_gaussian = torch.einsum('gb, mab->gma', pp, self.shape_function[1])
        # shape1_gaussian[g, m, a] = p_m(ξ_g) · C_{am}  (intermediate for ∂N_a/∂ξ)
        for i in range(self.num_nodes_per_elem):
            # J_{geij} += ∂N_a/∂ξ_j|_{ξ_g} · x_i^a
            Jacobian  += torch.einsum('gm,ei->geim', shape1_gaussian[:, :, i],
                                    nodes[elem_now[:, i]])

        # —— Step 3: det(J) and inverse J^{-1} ——
        det_Jacobian = Jacobian.det()
        inv_Jacobian = Jacobian.inverse()

        # —— Step 4: push-forward ∂N_a/∂x = J^{-1} · ∂N_a/∂ξ ——
        # ∂N_a/∂x_i = (J^{-1})_{ij} · ∂N_a/∂ξ_j
        self.shape_function_d1_gaussian = torch.einsum('gemi,gma->geia', inv_Jacobian, shape1_gaussian)

        # N_a(ξ) at Gaussian points: N_a = C_{am} · p_m(ξ_g)
        self.shape_function_d0_gaussian = torch.einsum('ab, gb->ga', self.shape_function[0],
                                    pp).unsqueeze(1)

        # —— Step 5: integration weight in physical space ——
        # w_g = w_g^ref · det(J),  dΩ = det(J) dξ
        self.gaussian_weight = torch.einsum('ge, g->ge', det_Jacobian, self.gaussian_weight_ref)

        # —— Step 6: second derivative of shape function at Gaussian points ——
        # shape2_gaussian[g, i, j, a] = ∂²N_a/∂ξ_i∂ξ_j|_{ξ_g}
        #   = C_{am} · ∂²p_m/∂ξ_i∂ξ_j|_{ξ_g}  =  p_m(ξ_g) · shape2_now[i, j, a, m]
        shape2_gaussian = torch.einsum('gb,mnab->gmna', pp, self.shape_function[2])

        # —— Step 7: second derivative of isoparametric mapping ——
        # Jacobian2: H_{ijk} = ∂²x_i/∂ξ_j∂ξ_k = Σ_a (∂²N_a/∂ξ_j∂ξ_k) · x_i^a
        #   second derivative of the isoparametric mapping
        Jacobian2 = torch.zeros([self._num_gaussian, len(self._elems), 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            # Jacobian2[g, e, i, j, k] += ∂²N_a/∂ξ_j∂ξ_k|_{ξ_g} · x_i^a
            Jacobian2 += torch.einsum('gmn,ei->geimn', shape2_gaussian[:, :, :, i],
                                        nodes[elem_now[:, i]])

        # inv_Jacobian2: ∂(J^{-1})_{ij}/∂x_k
        #   differentiate J·J^{-1}=I  →  ∂J^{-1}/∂x = -J^{-1}·(∂J/∂x)·J^{-1}
        #   in tensor index form:
        #   ∂(J^{-1})_{ml}/∂x_k = -(J^{-1})_{mj}·(J^{-1})_{pk}·(J^{-1})_{nl}·(∂²x_p/∂ξ_j∂ξ_n)·(J^{-1})_{?}
        #   simplified via chain rule → -J^{-1}·J^{-1}·J^{-1}·H
        inv_Jacobian2 = -torch.einsum(
            'gemj,gepk,genl,gejnp->gemlk', inv_Jacobian,
            inv_Jacobian, inv_Jacobian, Jacobian2)

        # —— Step 8: second derivative in physical space ∂²N_a/∂x_i∂x_j ——
        # Chain rule for second derivatives:
        #   ∂²N_a/∂x_i∂x_j = (J^{-1})_{im}·(J^{-1})_{jn}·(∂²N_a/∂ξ_m∂ξ_n)
        #                   + ∂(J^{-1})_{im}/∂x_j · (∂N_a/∂ξ_m)
        #
        # Term 1 (from isoparametric mapping of Hessian):
        #   [g, e, i, j, a] += (J^{-1})_{im}·(J^{-1})_{jn} · ∂²N_a/∂ξ_m∂ξ_n
        # Term 2 (correction from the derivative of J^{-1}):
        #   [g, e, i, j, a] += ∂(J^{-1})_{im}/∂x_j · ∂N_a/∂ξ_m
        self.shape_function_d2_gaussian = torch.einsum(
                'gemi, genj,gmna->geija',
                inv_Jacobian, inv_Jacobian, shape2_gaussian) + torch.einsum(
                    'gemij, gma->geija', inv_Jacobian2, shape1_gaussian)


        self._dNW = torch.einsum('geia,ge->geia',
                                self.shape_function_d1_gaussian,
                                self.gaussian_weight)
        
        self._dNdNW = torch.einsum('gelb,geia,ge->gelbia',
                                  self.shape_function_d1_gaussian,
                                  self.shape_function_d1_gaussian,
                                  self.gaussian_weight)


    @staticmethod
    def _shape_function_derivative(shape_function: torch.Tensor, ind: int):
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
    
    def _get_interpolation_coordinates(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Generate interpolation coordinates for shape functions.
        This method constructs a matrix of polynomial terms used for shape function interpolation
        in a 3D element. It builds terms based on the shape function's complexity, supporting
        constant, linear, quadratic, and cubic terms along with mixed terms.

        Args:
            nodes(torch.Tensor):
                Gaussian integration points with shape [num_gaussian, 3],
                containing the (x,y,z) coordinates of each point.

        Returns:
            torch.Tensor: 
                Matrix of polynomial terms with shape [num_gaussian, num_terms],
                where num_terms depends on the polynomial order of the shape functions:
                - 4 terms for linear (constant + x, y, z)
                - 7 terms for bilinear (adds xy, yz, zx)
                - 10 terms for quadratic (adds x², y², z²)
                - 17 terms for cubic without full terms (adds mixed quadratic terms + xyz)
                - 20 terms for full cubic (adds x³, y³, z³)
        """
        

        pp = torch.zeros([self._num_gaussian, self.shape_function[0].shape[1]], device=nodes.device)
        pp[:, 0] = 1
        pp[:, 1] = nodes[:, 0]
        pp[:, 2] = nodes[:, 1]
        pp[:, 3] = nodes[:, 2]
        if self.shape_function[0].shape[1] > 4:
            pp[:, 4] = nodes[:, 0] * nodes[:, 1]
            pp[:, 5] = nodes[:, 1] * nodes[:, 2]
            pp[:, 6] = nodes[:, 2] * nodes[:, 0]
        if self.shape_function[0].shape[1] > 7:
            pp[:, 7] = nodes[:, 0]**2
            pp[:, 8] = nodes[:, 1]**2
            pp[:, 9] = nodes[:, 2]**2
        if self.shape_function[0].shape[1] > 10:
            pp[:, 10] = nodes[:, 0]**2 * nodes[:, 1]
            pp[:, 11] = nodes[:, 1]**2 * nodes[:, 0]
            pp[:, 12] = nodes[:, 1]**2 * nodes[:, 2]
            pp[:, 13] = nodes[:, 2]**2 * nodes[:, 1]
            pp[:, 14] = nodes[:, 2]**2 * nodes[:, 0]
            pp[:, 15] = nodes[:, 0]**2 * nodes[:, 2]
            pp[:, 16] = nodes[:, 0] * nodes[:, 1] * \
                        nodes[:, 2]
        if self.shape_function[0].shape[1] > 17:
            pp[:, 17] = nodes[:, 0]**3
            pp[:, 18] = nodes[:, 1]**3
            pp[:, 19] = nodes[:, 2]**3
        
        return pp

    def get_gaussian_points(self, nodes: torch.Tensor):
        """
        Get the physical coordinates of the Gaussian points for the element.

        Args:
            nodes: [p, 3], global nodal coordinates x_i^a in physical space

        Returns:
            torch.Tensor: [num_gaussian, num_elem, 3], physical coordinates of Gaussian points for each element
        """
        pp = self._get_interpolation_coordinates(self.gaussian_coordinates)
        shapeFun0 = torch.einsum('ab, gb->ga', self.shape_function[0],
                                      pp)
        gaussian_position = torch.zeros(
            [self._num_gaussian, self._elems.shape[0], 3])
        for i in range(self._elems.shape[1]):
            gaussian_position = gaussian_position + torch.einsum(
                'g,eI->geI', shapeFun0[:, i], nodes[self._elems[:,
                                                                         i]])
        return gaussian_position.to(nodes.device)
    
    def get_mass_matrix(self,rotation_matrix:torch.Tensor=None):
        """
        Assemble the consistent mass matrix for the element.
        Returns:
            indices_force: torch.Tensor, indices for the force vector (flattened)
            Melement: torch.Tensor, element mass vector (flattened)
            indices_matrix: torch.Tensor, indices for the mass matrix (COO format)
            values: torch.Tensor, values for the global mass matrix (flattened)
        """
        # Consistent mass matrix: M_ij = ∫_Ω ρ N_i N_j dΩ
        # For each element, integrate N_i * N_j over the domain using Gaussian quadrature

        # shape_function_d0_gaussian: [num_gauss, num_elem, num_nodes_per_elem]
        N = self.shape_function_d0_gaussian  # [g, e, a]
        rho = self.density

        # Compute element mass matrix at each Gaussian point: [g, e, a, b]
        # M_ij = ∑_g N_i(g) * N_j(g) * w_g * detJ_g * ρ
        M_elem = torch.einsum('gea,geb,ge->abe', N, N, self.gaussian_weight * rho)

        # Expand to 3D (for vector-valued DoFs): [e, a, b] -> [a, j, b, k, e]
        # Only diagonal blocks are nonzero for lumped mass (consistent mass: block diagonal)
        num_elems = M_elem.shape[2]
        num_nodes = self.num_nodes_per_elem
        M_elem_full = torch.zeros([num_nodes, 3, num_nodes, 3, num_elems], device=M_elem.device, dtype=M_elem.dtype)
        for d in range(3):
            M_elem_full[:, d, :, d, :] = M_elem  # [a, b, e]

        # consider the rotation of the instance
        if rotation_matrix is not None:
            M_elem_full = torch.einsum('mj,ajbke,nk->ambne', rotation_matrix, M_elem_full, rotation_matrix)

        # Assemble into global matrix (same pattern as stiffness)
        values = torch.zeros([self._indices_matrix.shape[1]], device=M_elem.device, dtype=M_elem.dtype)
        values = values.scatter_add(0, self._index_matrix_coalesce, M_elem_full.flatten())

        return self._indices_matrix, values
        
    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)


        Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                         self.shape_function_d1_gaussian[:, :, :, i],
                                         U[self._elems[:, i]])

        F = Ugrad.clone()
        F[:, :, 0, 0] += 1
        F[:, :, 1, 1] += 1
        F[:, :, 2, 2] += 1

        W = torch.zeros([self._num_gaussian, self._elems.shape[0]], device=F.device, dtype=F.dtype)
        for mat_now in self._iter_material_values():
            W = W + mat_now.strain_energy_density_C3(F=F,)
        
        Ea = torch.einsum(
            'ge,ge->',W,
            self.gaussian_weight)

        return Ea
    
    def _get_EpdUe_EpdUe2(self, U: torch.Tensor, if_onlyforce: bool = False):
        """
        Calculate the first and second derivatives of the potential energy with respect to the nodal displacements U.
        Which are the residual force and the stiffness matrix of the element, respectively.

        Args:
            U: [num_nodes, 3], the nodal displacements
            if_onlyforce: whether to only calculate the residual force, if True, only return the residual force, otherwise return both the residual force and the stiffness matrix
        
        Returns:
            If if_onlyforce is True, returns the residual force as a flattened tensor.
            Otherwise, returns a tuple of the residual force and the stiffness matrix.

        Relement = ∂Ea/∂U with shape [num_nodes_per_elem, 3, num_elems]
        Ka_element = ∂²Ea/∂U² with shape [num_nodes_per_elem, 3, num_nodes_per_elem, 3, num_elems]
        """
        
        s, C = self.components_Solid(U=U)

        # calculate the element residual force
        Relement = torch.einsum('geij,geia->aje', s,
                                self._dNW)
        
        if if_onlyforce:
            return Relement
        
        # calculate the element tangential stiffness matrix
        Ka_element = torch.einsum('geijkl,gelbia->ajbke',
                                   C,
                                  self._dNdNW)
        
        return Relement, Ka_element
        

    def structural_Force(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None, if_onlyforce: bool = False):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)

        result = self._get_EpdUe_EpdUe2(U=U, if_onlyforce=if_onlyforce)
        
        
        if if_onlyforce:
            Relement = result
            if rotation_matrix is not None:
                Relement = torch.einsum('mj,aje->ame', rotation_matrix, Relement)
            return self._indices_force, Relement.flatten()

        
        if rotation_matrix is not None:
            Relement = torch.einsum('mj,aje->ame', rotation_matrix, result[0])
            Ka_element = torch.einsum('mj,ajbke,nk->ambne', rotation_matrix, result[1], rotation_matrix)
        else:
            Relement = result[0]
            Ka_element = result[1]
        
        # assembly the stiffness matrix and residual force                 
        values = torch.zeros([self._indices_matrix.shape[1]]).scatter_add(0, self._index_matrix_coalesce, Ka_element.flatten())
        
        return self._indices_force, Relement.flatten(), self._indices_matrix, values

    def components_Solid(self, U: torch.Tensor):
        Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad += torch.einsum('gki,kI->gkIi',
                                         self.shape_function_d1_gaussian[:, :, :, i],
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
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3], device=F.device, dtype=F.dtype)

        for mat_now in self._iter_material_values():
            s_now, C_now = mat_now.material_Constitutive_C3(
                F=F,
                J=J,
                Jneg=Jneg,
                invF=invF,
                I1=I1,
            )
            s = s + s_now
            C = C + C_now

        return s, C

    def get_volumn(self, U: torch.Tensor = None):
        if U is None:
            return self.gaussian_weight.sum()
        else:
            Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
            for i in range(self.num_nodes_per_elem):
                Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                            self.shape_function_d1_gaussian[:, :, :, i],
                                            U[self._elems[:, i]])
            F = Ugrad.clone()
            F[:, :, 0, 0] += 1
            F[:, :, 1, 1] += 1
            F[:, :, 2, 2] += 1
            J = F.det()
 
            return (self.gaussian_weight * J).sum()

    def set_required_DoFs(
            self, RGC_remain_index: np.ndarray) -> np.ndarray:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self._elems.unique().cpu().numpy()] = True

        return RGC_remain_index
    
    # region second order methods
    
    def get_2nd_order_point_index_surface(self, surface_ind: int) -> torch.Tensor:
        """
        The relative point index of the element that lies in the middle of the element

        get the 2-nd order point index of the element that lies in the middle of the element
        only for the first order faces of the second order element
        
        Args:
            surface_ind: the index of the surface, 0 for the first surface, 1 for the second surface, etc.
        
        Returns:
            torch.Tensor: the 2-nd order point index of the element \n
                size: [point_index, 3]\n
                    [0]: the index of the middle node of the element\n
                    [1]: the index of the neighbor node of the middle node of the element\n
                    [2]: the index of the other neighbor node of the middle node of the element\n
        """
        return torch.zeros([0, 3], dtype=torch.int64)
    
    
    # endregion second order methods


#========= Source code for Serializable.BaseElement.Element_3D.C3D8 =========#
class C3D8(Element_3D):
    """
    C3D8 - 8-node linear brick, full integration
    
    Local coordinates: g, h, r ∈ [-1, 1]
        origin: element center
    
    Node numbering (Abaqus convention):
        Bottom face (r=-1):  0(-1,-1,-1)  1( 1,-1,-1)  2( 1, 1,-1)  3(-1, 1,-1)
        Top face    (r= 1):  4(-1,-1, 1)  5( 1,-1, 1)  6( 1, 1, 1)  7(-1, 1, 1)
            
    Face definitions:
        face0: 0321 (Bottom, r=-1)    face1: 4567 (Top, r=1)
        face2: 0154 (Left,  g=-1)    face3: 1265 (Right, g=1)
        face4: 2376 (Front, h=-1)    face5: 0473 (Back, h=1)

    Shape functions:
        N_i = 1/8 (1 + g·g_i)(1 + h·h_i)(1 + r·r_i)
    """

    # ---- class-level static attributes (same pattern as tetrahedral.py / wedge.py) ----

    # Trilinear shape function coefficients in polynomial basis:
    #   [1, g, h, r, g*h, h*r, r*g, ..., g*h*r]
    shape_function = [
        torch.tensor([
            [ 0.125, -0.125, -0.125, -0.125,  0.125,  0.125,  0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0.125,  0.,  0.,  0.],
            [ 0.125,  0.125, -0.125, -0.125, -0.125,  0.125, -0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.125,  0.,  0.,  0.],
            [ 0.125,  0.125,  0.125, -0.125,  0.125, -0.125, -0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0.125,  0.,  0.,  0.],
            [ 0.125, -0.125,  0.125, -0.125, -0.125, -0.125,  0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.125,  0.,  0.,  0.],
            [ 0.125, -0.125, -0.125,  0.125,  0.125, -0.125, -0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.125,  0.,  0.,  0.],
            [ 0.125,  0.125, -0.125,  0.125, -0.125, -0.125,  0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0.125,  0.,  0.,  0.],
            [ 0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.125,  0.,  0.,  0.],
            [ 0.125, -0.125,  0.125,  0.125, -0.125,  0.125, -0.125,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0.125,  0.,  0.,  0.],
        ]),
    ]

    num_nodes_per_elem = 8
    num_surfaces = 6
    _num_gaussian = 8

    # Gauss-Legendre 2×2×2: weights = 1, points = ±1/√3
    gaussian_weight_ref = torch.ones(8)
    _p = 1.0 / np.sqrt(3.0)
    gaussian_coordinates = torch.tensor([
        [-_p, -_p, -_p],
        [ _p, -_p, -_p],
        [ _p,  _p, -_p],
        [-_p,  _p, -_p],
        [-_p, -_p,  _p],
        [ _p, -_p,  _p],
        [ _p,  _p,  _p],
        [-_p,  _p,  _p],
    ])

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        """
        Find the surface elements for a given surface index and element indices.
        
        Args:
            surface_ind: Surface index (0-5)
            elems_ind: Element indices
            
        Returns:
            torch.Tensor: Surface element node indices
        """
        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]

        if index_now.shape[0] == 0:
            quad_elems = torch.empty([0, 4],
                               dtype=torch.long,
                               device=self._elems.device)
            return [initialize_surfaces(quad_elems)]

        # Return appropriate face nodes according to face definitions in comments
        if surface_ind == 0:  # Bottom face (r=-1): face0: 0321
            quad_elems = self._elems[index_now][:, [0, 3, 2, 1]]
        elif surface_ind == 1:  # Top face (r=1): face1: 4567
            quad_elems = self._elems[index_now][:, [4, 5, 6, 7]]
        elif surface_ind == 2:  # Left face (g=-1): face2: 0154
            quad_elems = self._elems[index_now][:, [0, 1, 5, 4]]
        elif surface_ind == 3:  # Right face (g=1): face3: 1265
            quad_elems = self._elems[index_now][:, [1, 2, 6, 5]]
        elif surface_ind == 4:  # Front face (h=-1): face4: 2376
            quad_elems = self._elems[index_now][:, [2, 3, 7, 6]]
        elif surface_ind == 5:  # Back face (h=1): face5: 0473
            quad_elems = self._elems[index_now][:, [0, 4, 7, 3]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")

        return [initialize_surfaces(quad_elems)]


#========= Source code for Serializable.BaseElement.Element_3D.C3D8.C3D8R =========#
class C3D8R(C3D8):
    """
    C3D8R - 8-node linear brick, reduced integration with hourglass control
    
    Uses the same trilinear shape functions as C3D8, but with 1-point
    reduced integration (Gauss point at element center) plus hourglass
    stabilization based on the Flanagan-Belytschko algorithm.
    """

    # Override integration: single Gauss point at element center
    _num_gaussian = 1
    gaussian_weight_ref = torch.tensor([8.0])   # volume in ξ-space = 2³
    gaussian_coordinates = torch.tensor([[0.0, 0.0, 0.0]])

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

        # Aggregate shear modulus from all materials (parallel contribution assumption)
        shear_modulus = None
        for mat_now in self._iter_material_values():
            if not hasattr(mat_now, 'mu'):
                raise AttributeError(
                    f"Material {mat_now.__class__.__name__} has no attribute 'mu' required by hourglass control"
                )
            mu_now = mat_now.mu.flatten()
            shear_modulus = mu_now if shear_modulus is None else (shear_modulus + mu_now)

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


#========= Source code for Serializable.BaseElement.Element_3D.C3D20 =========#
class C3D20(Element_3D):
    """
    C3D20 - 20-node quadratic brick element (serendipity)
    
    Local coordinates: g, h, r ∈ [-1, 1], origin at element center.
    
    Node numbering (Abaqus convention):
        Bottom face (r=-1):  0-3-2-1  (corners),  11,10,9,8 (mid-edge)
        Top face    (r= 1):  4-5-6-7  (corners),  12,13,14,15 (mid-edge)
        Middle r=0  edges:   16(-1,-1,0)  17(1,-1,0)  18(1,1,0)  19(-1,1,0)
    
    Face definitions:
        face0: 0,3,2,1,11,10,9,8   (Bottom, r=-1)
        face1: 4,5,6,7,12,13,14,15 (Top, r=1)
        face2: 0,1,5,4,8,17,12,16  (Front, h=-1)
        face3: 1,2,6,5,9,18,13,17  (Right, g=1)
        face4: 2,3,7,6,10,19,14,18 (Back, h=1)
        face5: 0,4,7,3,16,15,19,11 (Left, g=-1)
    """

    # ---- class-level static attributes ----
    # Quadratic serendipity shape function coefficients (20 nodes × 20 basis terms)
    # Basis: [1, g, h, r, gh, hr, rg, g², h², r², g²h, gh², h²r, hr², r²g, rg², ghr, g²hr, gh²r, ghr²]
    shape_function = [
        torch.tensor([
            [-0.25,  0.125,  0.125,  0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125,  0.,     0.,     0.   ],
            [-0.25, -0.125,  0.125,  0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125, -0.125,  0.125, -0.125, -0.125,  0.125, -0.125,  0.125,  0.,     0.,     0.   ],
            [-0.25, -0.125, -0.125,  0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125,  0.125,  0.125, -0.125,  0.125,  0.125, -0.125, -0.125,  0.,     0.,     0.   ],
            [-0.25,  0.125, -0.125,  0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125,  0.125, -0.125, -0.125,  0.125, -0.125, -0.125,  0.125,  0.,     0.,     0.   ],
            [-0.25,  0.125,  0.125, -0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125, -0.125, -0.125,  0.125, -0.125, -0.125,  0.125,  0.125,  0.,     0.,     0.   ],
            [-0.25, -0.125,  0.125, -0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125, -0.125,  0.125,  0.125, -0.125,  0.125,  0.125, -0.125,  0.,     0.,     0.   ],
            [-0.25, -0.125, -0.125, -0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.125,  0.,     0.,     0.   ],
            [-0.25,  0.125, -0.125, -0.125,  0.,     0.,     0.,     0.125,  0.125,  0.125,  0.125, -0.125,  0.125,  0.125, -0.125,  0.125, -0.125,  0.,     0.,     0.   ],
            [ 0.25,  0.,    -0.25,  -0.25,   0.,     0.25,   0.,    -0.25,   0.,     0.,     0.25,   0.,     0.,     0.,     0.,     0.25,   0.,     0.,     0.,     0.   ],
            [ 0.25,  0.25,   0.,    -0.25,   0.,     0.,    -0.25,   0.,    -0.25,   0.,     0.,    -0.25,   0.25,   0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.,     0.25,  -0.25,   0.,    -0.25,   0.,    -0.25,   0.,     0.,    -0.25,   0.,     0.,     0.,     0.,     0.25,   0.,     0.,     0.,     0.   ],
            [ 0.25, -0.25,   0.,    -0.25,   0.,     0.,     0.25,   0.,    -0.25,   0.,     0.,     0.25,   0.25,   0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.,    -0.25,   0.25,   0.,    -0.25,   0.,    -0.25,   0.,     0.,     0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,     0.   ],
            [ 0.25,  0.25,   0.,     0.25,   0.,     0.,     0.25,   0.,    -0.25,   0.,     0.,    -0.25,  -0.25,   0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.,     0.25,   0.25,   0.,     0.25,   0.,    -0.25,   0.,     0.,    -0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,     0.   ],
            [ 0.25, -0.25,   0.,     0.25,   0.,     0.,    -0.25,   0.,    -0.25,   0.,     0.,     0.25,  -0.25,   0.,     0.,     0.,     0.,     0.,     0.,     0.   ],
            [ 0.25, -0.25,  -0.25,   0.,     0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,     0.25,   0.25,   0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.25,  -0.25,   0.,    -0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,     0.25,  -0.25,   0.,     0.,     0.,     0.,     0.   ],
            [ 0.25,  0.25,   0.25,   0.,     0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,    -0.25,  -0.25,   0.,     0.,     0.,     0.,     0.   ],
            [ 0.25, -0.25,   0.25,   0.,    -0.25,   0.,     0.,     0.,     0.,    -0.25,   0.,     0.,     0.,    -0.25,   0.25,   0.,     0.,     0.,     0.,     0.   ],
        ]),
    ]

    num_nodes_per_elem = 20
    num_surfaces = 6
    _num_gaussian = 27   # 3×3×3

    # Gauss-Legendre 3×3×3 weights (product of 1D weights)
    _w1d = torch.tensor([5.0/9.0, 8.0/9.0, 5.0/9.0])
    _x1d = torch.tensor([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gaussian_weight_ref = torch.ones(27)  # filled below
    gaussian_coordinates = torch.zeros([27, 3])  # filled below

    # Pre-compute the 3D Gauss points and weights
    _idx = 0
    for _i in range(3):
        for _j in range(3):
            for _k in range(3):
                gaussian_weight_ref[_idx] = _w1d[_i] * _w1d[_j] * _w1d[_k]
                gaussian_coordinates[_idx, 0] = _x1d[_i]
                gaussian_coordinates[_idx, 1] = _x1d[_j]
                gaussian_coordinates[_idx, 2] = _x1d[_k]
                _idx += 1
    del _idx, _i, _j, _k, _w1d, _x1d   # cleanup temporary loop variables

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        """
        Find surface elements for this element type
        
        Args:
            surface_ind: Surface index (0-5)
            elems_ind: Element indices to find surfaces for
            
        Returns:
            Tensor with surface node indices
        """
        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]
        
        if index_now.shape[0] == 0:
            quad_elems = torch.empty([0, 8], dtype=torch.long, device=self._elems.device)
            return [initialize_surfaces(quad_elems)]

        # Return appropriate face nodes according to face definitions in comments
        if surface_ind == 0:
            # Bottom face: 0-3-2-1 (nodes 0,3,2,1,11,10,9,8)
                quad_elems = self._elems[index_now][:, [0, 3, 2, 1, 11, 10, 9, 8]]
        elif surface_ind == 1:
            # Top face: 4-5-6-7 (nodes 4,5,6,7,12,13,14,15)
                quad_elems = self._elems[index_now][:, [4, 5, 6, 7, 12, 13, 14, 15]]
        elif surface_ind == 2:
            # Front face: 0-1-5-4 (nodes 0,1,5,4,8,17,12,16)
                quad_elems = self._elems[index_now][:, [0, 1, 5, 4, 8, 17, 12, 16]]
        elif surface_ind == 3:
            # Right face: 1-2-6-5 (nodes 1,2,6,5,9,18,13,17)
                quad_elems = self._elems[index_now][:, [1, 2, 6, 5, 9, 18, 13, 17]]
        elif surface_ind == 4:
            # Back face: 2-3-7-6 (nodes 2,3,7,6,10,19,14,18)
                quad_elems = self._elems[index_now][:, [2, 3, 7, 6, 10, 19, 14, 18]]
        elif surface_ind == 5:
            # Left face: 0-4-7-3 (nodes 0,4,7,3,16,15,19,11)
                quad_elems = self._elems[index_now][:, [0, 4, 7, 3, 16, 15, 19, 11]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
        
        return [initialize_surfaces(quad_elems)]

    def get_2nd_order_point_index_surface(self, surface_ind: int) -> torch.Tensor:
        """
        Get the 2nd order point index for the specified surface.
        This is used to identify the mid-edge nodes for the surface elements.
        
        Args:
            surface_ind: Surface index (0-5)
            
        Returns:
            torch.Tensor: Mid-edge node indices and their neighboring corner nodes
                size: [point_index, 3]
                [0]: the index of the middle node of the element
                [1]: the index of the neighbor node of the middle node of the element
                [2]: the index of the other neighbor node of the middle node of the element
        """
        if surface_ind == 0:
            # Bottom face: 0-3-2-1 with mid-edges 11,10,9,8
            return torch.tensor([[11, 0, 3],  # mid-edge between 0-3
                                [10, 2, 3],  # mid-edge between 3-2
                                [9, 1, 2],   # mid-edge between 2-1
                                [8, 0, 1]], dtype=torch.long, device='cpu')  # mid-edge between 1-0
        elif surface_ind == 1:
            # Top face: 4-5-6-7 with mid-edges 12,13,14,15
            return torch.tensor([[12, 4, 5],  # mid-edge between 4-5
                                [13, 5, 6],  # mid-edge between 5-6
                                [14, 6, 7],  # mid-edge between 6-7
                                [15, 4, 7]], dtype=torch.long, device='cpu')  # mid-edge between 7-4
        elif surface_ind == 2:
            # Front face: 0-1-5-4 with mid-edges 8,17,12,16
            return torch.tensor([[8, 0, 1],   # mid-edge between 0-1
                                [17, 1, 5],  # mid-edge between 1-5
                                [12, 4, 5],  # mid-edge between 5-4
                                [16, 0, 4]], dtype=torch.long, device='cpu')  # mid-edge between 4-0
        elif surface_ind == 3:
            # Right face: 1-2-6-5 with mid-edges 9,18,13,17
            return torch.tensor([[9, 1, 2],   # mid-edge between 1-2
                                [18, 2, 6],  # mid-edge between 2-6
                                [13, 5, 6],  # mid-edge between 6-5
                                [17, 1, 5]], dtype=torch.long, device='cpu')  # mid-edge between 5-1
        elif surface_ind == 4:
            # Back face: 2-3-7-6 with mid-edges 10,19,14,18
            return torch.tensor([[10, 2, 3],  # mid-edge between 2-3
                                [19, 3, 7],  # mid-edge between 3-7
                                [14, 6, 7],  # mid-edge between 7-6
                                [18, 2, 6]], dtype=torch.long, device='cpu')  # mid-edge between 6-2
        elif surface_ind == 5:
            # Left face: 0-4-7-3 with mid-edges 16,15,19,11
            return torch.tensor([[16, 0, 4],  # mid-edge between 0-4
                                [15, 4, 7],  # mid-edge between 4-7
                                [19, 3, 7],  # mid-edge between 7-3
                                [11, 0, 3]], dtype=torch.long, device='cpu')  # mid-edge between 3-0
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")


#========= Source code for Serializable.BaseElement.Element_3D.C3D6 =========#
class C3D6(Element_3D):
    """
    # Local coordinates:
        origin: 0-th nodal
        ksi_0: 0-1 vector
        ksi_1: 0-2 vector
        ksi_2: 0-3 vector

    # face nodal always point at the void
        face0: 021 (Triangle)
        face1: 345 (Triangle)
        face2: 0143 (Rectangle)
        face3: 1254 (Rectangle)
        face4: 2035 (Rectangle)
    
    # shape_funtion:
        N_0 = 0.5 * (1 - ksi_0 - ksi_1) * (1 - ksi_2) \n
        N_1 = 0.5 * ksi_0 * (1 - ksi_2) \n
        N_2 = 0.5 * ksi_1 * (1 - ksi_2) \n
        N_3 = 0.5 * (1 - ksi_0 - ksi_1) * (1 + ksi_2) \n
        N_4 = 0.5 * ksi_0 * (1 + ksi_2) \n
        N_5 = 0.5 * ksi_1 * (1 + ksi_2) \n
    """
    shape_function = [
        torch.tensor([
            [0.5, -0.5, -0.5, -0.5, 0.0, 0.5, 0.5],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0],
            [0.5, -0.5, -0.5, 0.5, 0.0, -0.5, -0.5],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
        ]),
    ]
    num_nodes_per_elem = 6
    _num_gaussian = 2
    gaussian_weight_ref = torch.tensor([1 / 2, 1 / 2])
    gaussian_coordinates = torch.tensor([
        [1/3, 1/3, 1 / np.sqrt(3)],
        [1/3, 1/3, -1 / np.sqrt(3)],
    ])
    num_surfaces = 5

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]
        
        if index_now.shape[0] == 0:
            tri_elems = torch.empty([0, 3], dtype=torch.long, device=self._elems.device)
            return []
        
        if surface_ind in [0, 1]:
            if surface_ind == 0:
                tri_elems = self._elems[index_now][:, [0, 2, 1]]
            elif surface_ind == 1:
                tri_elems = self._elems[index_now][:, [3, 4, 5]]
            return [initialize_surfaces(tri_elems)]
        elif surface_ind in [2, 3, 4]:
            if surface_ind == 2:
                quad_elems = self._elems[index_now][:, [0, 1, 4, 3]]
            elif surface_ind == 3:
                quad_elems = self._elems[index_now][:, [1, 2, 5, 4]]
            elif surface_ind == 4:
                quad_elems = self._elems[index_now][:, [2, 0, 3, 5]]
            return [initialize_surfaces(quad_elems)]

        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")


#========= Source code for Serializable.BaseElement.Element_3D.C3D15 =========#
class C3D15(Element_3D):
    """
    # Local coordinates:
        origin: bottom triangle center
        g, h: coordinates in triangle base
        r: coordinate along prism height

    # Node numbering:
        - Bottom face (r=-1): 0, 1, 2 (vertices), 6, 7, 8 (mid-edge)
        - Top face (r=1): 3, 4, 5 (vertices), 9, 10, 11 (mid-edge)
        - Middle nodes (r=0): 12, 13, 14 (on vertical edges)

    # Face description:
        face0: 0(8)2(7)1(6) (Triangle)
        face1: 3(9)4(10)5(11) (Triangle)
        face2: 0(6)1(13)4(9)3(12) (Rectangle)
        face3: 1(7)2(14)5(10)4(13) (Rectangle)
        face4: 2(8)0(12)3(11)5(14) (Rectangle)

    # Shape functions:
        Quadratic interpolation in all directions
        Combines triangular base shape functions with prismatic extrusion
    """
    shape_function = [
        torch.tensor([
            [
                0, -1.0, -1.0, -0.5, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0, 0,
                -1.0, -0.5, -0.5, -1.0, -2.0, 0, 0, 0
            ],
            [
                0, -1.0, 0, 0, 0, 0, 0.5, 1.0, 0, 0, 0, 0, 0,
                0, 0.5, -1.0, 0, 0, 0, 0
            ],
            [
                0, 0, -1.0, 0, 0, 0.5, 0, 0, 1.0, 0, 0, 0,
                -1.0, 0.5, 0, 0, 0, 0, 0, 0
            ],
            [
                0, -1.0, -1.0, 0.5, 2.0, -1.5, -1.5, 1.0,
                1.0, 0.5, 0, 0, 1.0, -0.5, -0.5, 1.0, 2.0, 0,
                0, 0
            ],
            [
                0, -1.0, 0, 0, 0, 0, -0.5, 1.0, 0, 0, 0, 0,
                0, 0, 0.5, 1.0, 0, 0, 0, 0
            ],
            [
                0, 0, -1.0, 0, 0, -0.5, 0, 0, 1.0, 0, 0, 0,
                1.0, 0.5, 0, 0, 0, 0, 0, 0
            ],
            [
                0, 2.0, 0, 0, -2.0, 0, -2.0, -2.0, 0, 0, 0,
                0, 0, 0, 0, 2.0, 2.0, 0, 0, 0
            ],
            [
                0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, -2.0, 0, 0, 0
            ],
            [
                0, 0, 2.0, 0, -2.0, -2.0, 0, 0, -2.0, 0, 0,
                0, 2.0, 0, 0, 0, 2.0, 0, 0, 0
            ],
            [
                0, 2.0, 0, 0, -2.0, 0, 2.0, -2.0, 0, 0, 0, 0,
                0, 0, 0, -2.0, -2.0, 0, 0, 0
            ],
            [
                0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 2.0, 0, 0, 0
            ],
            [
                0, 0, 2.0, 0, -2.0, 2.0, 0, 0, -2.0, 0, 0, 0,
                -2.0, 0, 0, 0, -2.0, 0, 0, 0
            ],
            [
                1.0, -1.0, -1.0, 0, 0, 0, 0, 0, 0, -1.0, 0,
                0, 0, 1.0, 1.0, 0, 0, 0, 0, 0
            ],
            [
                0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                -1.0, 0, 0, 0, 0, 0
            ],
            [
                0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                -1.0, 0, 0, 0, 0, 0, 0
            ],
        ]),
    ]
    num_nodes_per_elem = 15
    _num_gaussian = 9
    gaussian_weight_ref = torch.tensor([
        5.0/54.0, 5.0/54.0, 5.0/54.0,
        8.0/54.0, 8.0/54.0, 8.0/54.0,
        5.0/54.0, 5.0/54.0, 5.0/54.0,
    ])
    gaussian_coordinates = torch.tensor([
        [1/6, 1/6, -np.sqrt(3 / 5)],
        [2/3, 1/6, -np.sqrt(3 / 5)],
        [1/6, 2/3, -np.sqrt(3 / 5)],
        [1/6, 1/6, 0.0],
        [2/3, 1/6, 0.0],
        [1/6, 2/3, 0.0],
        [1/6, 1/6, np.sqrt(3 / 5)],
        [2/3, 1/6, np.sqrt(3 / 5)],
        [1/6, 2/3, np.sqrt(3 / 5)],
    ])
    num_surfaces = 5

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]
        
        if index_now.shape[0] == 0:
            return []
        
        T6_elems = torch.empty([0, 6], dtype=torch.long, device=self._elems.device)
        quad_elems = torch.empty([0, 8], dtype=torch.long, device=self._elems.device)

        if surface_ind == 0:
            # Bottom triangular face: 0(8)2(7)1(6) -> T6 elements
            T6_elems = self._elems[index_now][:, [0, 2, 1, 8, 7, 6]]
            
        elif surface_ind == 1:
            # Top triangular face: 3(9)4(10)5(11) -> T6 elements

            T6_elems = self._elems[index_now][:, [3, 4, 5, 9, 10, 11]]
            
        elif surface_ind == 2:
            # Rectangular face: 0(6)1(13)4(9)3(12) -> Q8 elements


            quad_elems = self._elems[index_now][:, [0, 1, 4, 3, 6, 13, 9, 12]]
            
        elif surface_ind == 3:
            # Rectangular face: 1(7)2(14)5(10)4(13) -> Q8 elements


            quad_elems = self._elems[index_now][:, [1, 2, 5, 4, 7, 14, 10, 13]]
            
        elif surface_ind == 4:
            # Rectangular face: 2(8)0(12)3(11)5(14) -> Q8 elements

            quad_elems = self._elems[index_now][:, [2, 0, 3, 5, 8, 12, 11, 14]]
            
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
        
        result = []
        if T6_elems.shape[0] > 0:
            result.append(initialize_surfaces(T6_elems))
        if quad_elems.shape[0] > 0:
            result.append(initialize_surfaces(quad_elems))
        return result
    
    def get_2nd_order_point_index_surface(self, surface_ind: int):
        """
        Get the 2nd order point index for the specified surface.
        This is used to identify the mid-edge nodes for the surface elements.
        """
        if surface_ind == 0:
            return torch.tensor([[8, 0, 2],
                                    [7, 1, 2],
                                    [6, 0, 1]], dtype=torch.long, device='cpu')
        if surface_ind == 1:
            return torch.tensor([[9, 3, 4],
                                    [10, 4, 5],
                                    [11, 3, 5]], dtype=torch.long, device='cpu')
        if surface_ind == 2:
            return torch.tensor([[6, 0, 1],
                                    [13, 1, 4],
                                    [9, 3, 4],
                                    [12, 0, 3]], dtype=torch.long, device='cpu')
        if surface_ind == 3:
            return torch.tensor([[7, 1, 2],
                                    [14, 2, 5],
                                    [10, 4, 5],
                                    [13, 1, 4]], dtype=torch.long, device='cpu')
        if surface_ind == 4:
            return torch.tensor([[8, 0, 2],
                                    [12, 0, 3],
                                    [11, 3, 5],
                                    [14, 2, 5]], dtype=torch.long, device='cpu')
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")


#========= Source code for Serializable.BaseElement.Element_3D.C3D4 =========#
class C3D4(Element_3D):
    """
        Local coordinates:
            origin: 0-th nodal
            ksi_0: 0-1 vector
            ksi_1: 0-2 vector
            ksi_2: 0-3 vector

        face nodal always point at the void
            face0: 021
            face1: 013
            face2: 123
            face3: 032

        shape_funtion:
            N_i = ksi_i * ksi_i, i<=3
    """
    shape_function = [
        torch.tensor([[1.0, -1.0, -1.0, -1.0], [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    ]

    num_nodes_per_elem = 4
    _num_gaussian = 1
    
    gaussian_weight_ref = torch.tensor([1 / 6])

    gaussian_coordinates = torch.tensor([[0.25, 0.25, 0.25]])

    num_surfaces = 4

    
    def extract_surface(self, surface_ind: int,
                           elems_ind: torch.Tensor):

        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]

        if index_now.shape[0] == 0:
            tri_elems = torch.empty([0, 3], dtype=torch.long, device=self._elems.device)

        elif surface_ind == 0:
            tri_elems = self._elems[index_now][:, [0, 2, 1]]
        elif surface_ind == 1:
            tri_elems = self._elems[index_now][:, [0, 1, 3]]
        elif surface_ind == 2:
            tri_elems = self._elems[index_now][:, [1, 2, 3]]
        elif surface_ind == 3:
            tri_elems = self._elems[index_now][:, [0, 3, 2]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")

        return [initialize_surfaces(tri_elems)]


#========= Source code for Serializable.BaseElement.Element_3D.C3D10 =========#
class C3D10(Element_3D):
    """
        Local coordinates:
            origin: 0-th nodal
            ksi_0: 0-1 vector
            ksi_1: 0-2 vector
            ksi_2: 0-3 vector

        face nodal always point at the void
            face0: 0(6)2(5)1(4)
            face1: 0(4)1(8)3(7)
            face2: 1(5)2(9)3(8)
            face3: 0(7)3(9)2(6)

        2-nd element extra nodals:
            4(01) 5(12) 6(02) 7(03) 8(13) 9(23)

        shape_funtion:
            N_i = (2 ksi_i - 1) * ksi_i, i<=2 \n
            N_i = 4 ksi_j ksi_k, i>2 and jk is the neighbor nodals fo i-th nodal
    """
    shape_function = [
        torch.tensor([[1., -3., -3., -3., 4., 4., 4., 2., 2., 2.],
                      [0., -1., 0., 0., 0., 0., 0., 2., 0., 0.],
                      [0., 0., -1., 0., 0., 0., 0., 0., 2., 0.],
                      [0., 0., 0., -1., 0., 0., 0., 0., 0., 2.],
                      [0., 4., 0., 0., -4., 0., -4., -4., 0., 0.],
                      [0., 0., 0., 0., 4., 0., 0., 0., 0., 0.],
                      [0., 0., 4., 0., -4., -4., 0., 0., -4., 0.],
                      [0., 0., 0., 4., 0., -4., -4., 0., 0., -4.],
                      [0., 0., 0., 0., 0., 0., 4., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 4., 0., 0., 0., 0.]]),
    ]
    num_nodes_per_elem = 10
    _num_gaussian = 4
    gaussian_weight_ref = torch.tensor([1 / 24, 1 / 24, 1 / 24, 1 / 24])
    gaussian_coordinates = torch.tensor([
        [0.13819660, 0.13819660, 0.13819660],
        [0.58541020, 0.13819660, 0.13819660],
        [0.13819660, 0.58541020, 0.13819660],
        [0.13819660, 0.13819660, 0.58541020],
    ])
    num_surfaces = 4



    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]

        if index_now.shape[0] == 0:
            return []

        if surface_ind == 0:
            T6_elems = self._elems[index_now][:, [0, 2, 1, 6, 5, 4]]
        elif surface_ind == 1:
            T6_elems = self._elems[index_now][:, [0, 1, 3, 4, 8, 7]]
        elif surface_ind == 2:
            T6_elems = self._elems[index_now][:, [1, 2, 3, 5, 9, 8]]
        elif surface_ind == 3:
            T6_elems = self._elems[index_now][:, [0, 3, 2, 7, 9, 6]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")

        result = []
        if T6_elems.shape[0] > 0:
            result.append(initialize_surfaces(T6_elems))
        return result

    def get_2nd_order_point_index_surface(self, surface_ind: int):
        """
        Get the 2nd order point index for the specified surface.
        This is used to identify the mid-edge nodes for the surface elements.
        """
        if surface_ind == 0:
            return torch.tensor([[6, 0, 2], [5, 1, 2], [4, 0, 1]], dtype=torch.long)
        elif surface_ind == 1:
            return torch.tensor([[4, 0, 1], [8, 1, 3], [7, 0, 3]], dtype=torch.long)
        elif surface_ind == 2:
            return torch.tensor([[5, 1, 2], [9, 2, 3], [8, 1, 3]], dtype=torch.long)
        elif surface_ind == 3:
            return torch.tensor([[7, 0, 3], [9, 2, 3], [6, 0, 2]], dtype=torch.long)
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")


#========= Source code for Serializable.BaseObj =========#
class BaseObj(Serializable):

    def __init__(self) -> None:

        super().__init__()

        """
        Initialize the FEA_Obj_Base class.
        """
        self._RGC_requirements: list[int] = [0]
        """
        The number of required RGCs for this object.
        """

        self._RGC_index: int = None
        """
        The index of the extra RGC for this object.
        """

        self._index_start: int = None
        """
        The start index of the extra RGC for this object
        """

        self._assembly: Assembly = None
        """The assembly this object belongs to."""

    @property
    def serialized_attributes(self):
        """Get the list of attributes to be serialized."""
        serialized_attrs = super().serialized_attributes
        serialized_attrs = [attr for attr in serialized_attrs if attr != '_assembly']
        serialized_attrs += ['_RGC_requirements']

        return serialized_attrs

    def set_RGC_index(self, index: int) -> None:
        """
        Set the index of the extra RGC for this object.
        """
        self._RGC_index = index
        

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        return RGC_remain_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return RGC

    def initialize(self, assembly: Assembly):
        self._assembly = assembly
        self._index_start = assembly.RGC_list_indexStart[self._RGC_index]
        
    def initialize_dynamic(self):
        pass
    
    def reinitialize(self, RGC: list[torch.Tensor]):
        pass


#========= Source code for Serializable._Surfaces =========#
class _Surfaces(Serializable):
    """
    Class representing a set of surfaces in the finite element model.
    """

    _serialized_attributes = ['_surface_dict']

    def __init__(self):
        self._surface_dict: dict[str, list[tuple[np.ndarray, int]]] = {}
        self._surface_elements: dict[str, list[BaseSurface]] = {}
        self._initialized = False

    def initialize(self, part: Part):
        """
        Initialize the surface set before FEA.
        """
        self._surface_elements = {}
        for name, surface_indices in self._surface_dict.items():
            element_now = part.extract_surfaces(name)
            self._surface_elements[name] = element_now

            # initialize the surface elements
            for se in element_now:
                se.initialize(part)
                self._initialized = True

    def get_elements(self, name: str) -> list[BaseSurface]:
        """
        Get the surface elements by their name.

        Args:
            name (str): The name of the surface.

        Returns:
            list[BaseSurface]: The list of surface elements.
        """
        return self._surface_elements.get(name, [])
    
    def get_trimesh(self, name: str) -> torch.Tensor:
        """
        Get the triangles of a surface set by name.

        Args:
            name (str): Name of the surface set.

        Returns:
            torch.Tensor: Triangles of the surface set.
        """
        surface_elements = self.get_elements(name)
        if len(surface_elements) == 0:
            raise ValueError(f"Surface set '{name}' not found in the model.")
        
        triangles = []
        for se in surface_elements:
            triangles.append(se.trimesh)
        return torch.cat(triangles, dim=0)

    def __getitem__(self, key: str):
        """
        Get a surface by its name.

        Args:
            key (str): The name of the surface.

        Returns:
            list[tuple[np.ndarray, int]]: The surface data.
        """
        return self._surface_dict[key]

    def __setitem__(self, key: str, value: list[tuple[np.ndarray, int]]):
        """
        Set a surface by its name.

        Args:
            key (str): The name of the surface.
            value (list[tuple[np.ndarray, int]]): The surface data.
        """
        self._surface_dict[key] = value



    def __contains__(self, key: str):
        """
        Check if a surface exists by its name.

        Args:
            key (str): The name of the surface.

        Returns:
            bool: True if the surface exists, False otherwise.
        """
        return key in self._surface_dict

    def keys(self):
        """
        Get the keys of the surface set.

        Returns:
            list[str]: The list of surface names.
        """
        return list(self._surface_dict.keys())


#========= Source code for Serializable.Part =========#
class Part(Serializable):



    def __init__(self, nodes: torch.Tensor) -> None:

        self.nodes: torch.Tensor = nodes
        """
        Nodes of the part.
        Shape: (num_nodes, 3)
        """
        self.elems: dict[str, BaseElement] = {}
        """
        Elements of the part.
        """

        self.set_nodes: dict[str, np.ndarray] = {}
        """Set of nodes for the part."""

        self.surfaces = _Surfaces()
        """Surfaces of the part."""

        self.mass_matrix_indices: torch.Tensor = None
        """
        Indices of the mass matrix.
        """
        self.mass_matrix_values: torch.Tensor = None
        """
        Values of the mass matrix.
        """

        self.mid_pt_idxmap: dict[tuple[int, int], int] = {}
        """A mapping from edge (node index pair) to midpoint node index, used for quadratic element conversion."""

        self.mid_pt_idxmap_torch: torch.Tensor = None
        """Same mapping as mid_pt_idxmap but in torch.Tensor format for efficient lookup during FEA calculations.
        
        [0]: node index 0 of the edge
        [1]: node index 1 of the edge
        [2]: midpoint node index corresponding to the edge
        """

    def initialize(self):
        for e in self.elems.values():
            e.initialize(self.nodes)
        self.surfaces.initialize(self)

    # region CAD
    def add_element(self, element: BaseElement, name: str = None):
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
    
    def add_surface_set(self, name: str, elements: np.ndarray):
        """
        Add a surface set to the FEA model.
        
        Args:
            name (str): Name of the surface set.
            elements (np.ndarray): Surface elements information.
            
        Returns:
            str: Name of the added surface set.
        """
        self.surfaces[name] = elements
        return name
    
    def delete_surface_set(self, name: str):
        """
        Delete a surface set from the FEA model.
        
        Args:
            name (str): Name of the surface set to delete.
            
        Raises:
            KeyError: If the surface set doesn't exist.
        """
        if name in self.surfaces:
            del self.surfaces[name]
        else:
            raise KeyError(f"Surface set '{name}' not found in the model.")

    def refine_RGC(self, RGC: torch.Tensor) -> torch.Tensor:
        RGC_out = RGC
        for e in self.elems.values():
            RGC_out = e.refine_RGC(RGC_out, self.nodes)
        return RGC_out

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
            elems0 = self.elems[element_name_list[0]]
            self.add_element(elems0, name=element_name_new)
            self.delete_element(element_name_list[0])
            return
            
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

    def convert_linear_to_quadratic_elements(self, element_name_list: list[str], new_element_name_list: list[str], if_update_setnodes = True):
        """
        Convert selected linear elements into quadratic elements.

        Args:
            element_name_list (list[str]): Names of the existing linear elements.
            new_element_name_list (list[str]): Names for the new quadratic elements.
            

        This method replaces each old element in-place by a new quadratic element:
        1. Extract all unique edges from the selected linear elements.
        2. Create midpoint nodes for those edges and append them to part nodes.
        3. Rebuild element connectivity according to the quadratic element node order.
        4. Preserve density and material definitions.
        """
        if len(element_name_list) != len(new_element_name_list):
            raise ValueError('element_name_list and new_element_name_list must have the same length.')

        if len(set(new_element_name_list)) != len(new_element_name_list):
            raise ValueError('new_element_name_list contains duplicate names.')

        supported_conversion = {
            'C3D4': 'C3D10',
            'C3D6': 'C3D15',
            'C3D8': 'C3D20',
            'C3D8R': 'C3D20',
        }

        selected: list[tuple[str, str, BaseElement]] = []
        reserved_new_names = set(self.elems.keys())

        for old_name, new_name in zip(element_name_list, new_element_name_list):
            if old_name not in self.elems:
                raise ValueError(f"Element '{old_name}' not found in the model.")
            if new_name in reserved_new_names and new_name != old_name:
                raise ValueError(f"New element name '{new_name}' already exists in the model.")

            element = self.elems[old_name]
            old_type = element.__class__.__name__
            if old_type not in supported_conversion:
                raise ValueError(f"Unsupported linear element type '{old_type}'.")

            expected_new_type = supported_conversion[old_type]
            if new_name != expected_new_type and new_name != old_name:
                # We only verify the class type, not the text of the name,
                # because user may pass an arbitrary new element name.
                pass

            selected.append((old_name, new_name, element))

        edge_to_mid_index: dict[tuple[int, int], int] = {}
        node_count = self.nodes.shape[0]

        # extract all edges
        edges_all = []
        for _, _, element in selected:
            linear_edges_idx = self._get_linear_edges_for_element_type(element.__class__.__name__)
            elems = element._elems
            linear_edges = elems[:, linear_edges_idx].cpu().numpy()
            edges_all.append(linear_edges.reshape(-1, 2))
        edges_all = np.concatenate(edges_all, axis=0)
        edges_all = np.sort(edges_all, axis=1)  # sort node indices in each edge to avoid duplicates like (1,2) and (2,1)
        large_number = edges_all.max() + 1
        edge_hash = edges_all[:, 0] * large_number + edges_all[:, 1]
        _, unique_indices = np.unique(edge_hash, return_index=True)
        unique_edges = edges_all[unique_indices]

        # create the mapping from edge to midpoint node index
        for i, (n1, n2) in enumerate(unique_edges):
            edge_to_mid_index[(n1.item(), n2.item())] = node_count + i
            edge_to_mid_index[(n2.item(), n1.item())] = node_count + i  # add both directions for easy lookup

        # create mid nodes for unique edges
        new_node_tensor = self.nodes[unique_edges].mean(dim=1)
        self.nodes = torch.cat([self.nodes, new_node_tensor], dim=0)

        for old_name, new_name, element in selected:
            old_type = element.__class__.__name__
            new_type = supported_conversion[old_type]
            new_connectivity = self._build_quadratic_connectivity(old_type, element._elems, edge_to_mid_index)

            new_element = elements.initialize_element(
                element_type=new_type,
                elems_index=element._elems_index,
                elems=new_connectivity,
                part=self,
            )
            new_element.density = element.density
            if isinstance(element.materials, dict):
                new_element.materials = dict(element.materials)
            else:
                new_element.materials = element.materials

            if new_name == old_name:
                self.delete_element(old_name)
                self.add_element(new_element, name=new_name)
            else:
                self.delete_element(old_name)
                self.add_element(new_element, name=new_name)

        self.mid_pt_idxmap = edge_to_mid_index
        self.mid_pt_idxmap_torch = torch.tensor([[n1, n2, mid_idx] for (n1, n2), mid_idx in edge_to_mid_index.items()], dtype=torch.long)

        if if_update_setnodes:
            for key in self.set_nodes.keys():
                set_nodes = self.set_nodes[key]
                new_set_nodes = set(set_nodes.tolist())
                for n1, n2 in unique_edges:
                    if n1 in new_set_nodes and n2 in new_set_nodes:
                        mid_idx = edge_to_mid_index[(n1, n2)]
                        new_set_nodes.add(mid_idx)
                self.set_nodes[key] = np.sort(np.array(list(new_set_nodes)))

    def _get_linear_edges_for_element_type(self, element_type: str) -> list[list[int]]:
        if element_type == 'C3D4':
            return [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]]
        if element_type == 'C3D6':
            return [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5], [0, 3], [1, 4], [2, 5]]
        if element_type in ['C3D8', 'C3D8R']:
            return [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        raise ValueError(f"Unsupported element type '{element_type}' for edge extraction.")

    def _build_quadratic_connectivity(self, element_type: str, elems: torch.Tensor, edge_to_mid_index: dict[tuple[int, int], int]) -> torch.Tensor:
        def mid(i: int, j: int) -> torch.Tensor:
            edge0 = elems[:, i].cpu().numpy()
            edge1 = elems[:, j].cpu().numpy()
            mid_idx = torch.tensor([edge_to_mid_index[(n1, n2)] for n1, n2 in zip(edge0, edge1)], dtype=torch.long, device=elems.device)
            return mid_idx

        if element_type == 'C3D4':
            return torch.stack([
                elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3],
                mid(0, 1), mid(1, 2), mid(0, 2), mid(0, 3), mid(1, 3), mid(2, 3),
            ], dim=1)

        if element_type == 'C3D6':
            return torch.stack([
                elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3], elems[:, 4], elems[:, 5],
                mid(0, 1), mid(1, 2), mid(0, 2), mid(3, 4), mid(4, 5), mid(3, 5),
                mid(0, 3), mid(1, 4), mid(2, 5),
            ], dim=1)

        if element_type in ['C3D8', 'C3D8R']:
            return torch.stack([
                elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3],
                elems[:, 4], elems[:, 5], elems[:, 6], elems[:, 7],
                mid(0, 1), mid(1, 2), mid(2, 3), mid(3, 0),
                mid(4, 5), mid(5, 6), mid(6, 7), mid(7, 4),
                mid(0, 4), mid(1, 5), mid(2, 6), mid(3, 7),
            ], dim=1)

        raise ValueError(f"Unsupported element type '{element_type}' for quadratic connectivity.")

    def extract_surfaces(self, name: str) -> list[BaseSurface]:
        """        Get the triangles of a surface set by name.  
        
        Args:
            name (str): Name of the surface set.
            
        Returns:
            list[BaseSurface]: List of triangles in the surface set.
            
        Raises:
            ValueError: If the surface set is not found.
        """
        surface = []
        for surf_index in self.surfaces[name]:
            elem_ind = surf_index[0]
            surf_ind = surf_index[1]
            for e in self.elems.values():
                s_now = e.extract_surface(surf_ind, elem_ind)
                surface += s_now
        if len(surface) == 0:
            raise ValueError(f"Surface {surf_ind} not found in the model.")
        else:
            return surfaces.merge_surfaces(surface)

    def export_inp(self, file_path: str, part_name: str = 'Part-1') -> None:
        """
        Export the part to an Abaqus INP file.

        The generated INP includes:
        - node coordinates
        - element connectivity for all elements in the part
        - node sets from self.set_nodes
        - surfaces from self.surfaces
        """
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('** Generated by torchfea Part.export_inp\n')
            f.write(f'*Part, name={part_name}\n')

            # Write nodes
            f.write('*Node\n')
            for node_id, node in enumerate(self.nodes, start=1):
                coords = node.tolist()
                f.write(f'{node_id}, {coords[0]}, {coords[1]}, {coords[2]}\n')

            # Write elements grouped by Abaqus element type
            elements_by_type: dict[str, list[BaseElement]] = {}
            for element in self.elems.values():
                elem_type = element.__class__.__name__
                elements_by_type.setdefault(elem_type, []).append(element)

            for elem_type, element_list in elements_by_type.items():
                f.write(f'*Element, type={elem_type}\n')
                for element in element_list:
                    connectivity = element._elems
                    element_ids = element._elems_index
                    for elem_row, elem_id in zip(connectivity, element_ids):
                        node_ids = [str(int(node_id.item()) + 1) for node_id in elem_row]
                        f.write(f'{int(elem_id.item()) + 1}, ' + ', '.join(node_ids) + '\n')

            # Write node sets
            for set_name, node_list in self.set_nodes.items():
                sorted_nodes = sorted(np.asarray(node_list, dtype=int).tolist())
                if len(sorted_nodes) == 0:
                    continue
                f.write(f'*Nset, nset={set_name}\n')
                for i in range(0, len(sorted_nodes), 10):
                    line_nodes = sorted_nodes[i:i + 10]
                    f.write(', '.join(str(node_id + 1) for node_id in line_nodes) + '\n')

            # Write surfaces
            for surface_name in self.surfaces.keys():
                surface_items = self.surfaces[surface_name]
                if len(surface_items) == 0:
                    continue
                f.write(f'*Surface, type=ELEMENT, name={surface_name}\n')
                for elem_ids, surface_index in surface_items:
                    if isinstance(elem_ids, np.ndarray):
                        elem_ids_iter = elem_ids.ravel().tolist()
                    elif isinstance(elem_ids, (list, tuple)):
                        elem_ids_iter = list(elem_ids)
                    else:
                        elem_ids_iter = [int(elem_ids)]
                    for elem_id in elem_ids_iter:
                        f.write(f'{int(elem_id) + 1}, S{int(surface_index) + 1}\n')

            f.write('*End Part\n')

    # endregion

    # region FEA

    def potential_energy(self, RGC: torch.Tensor, rotation_matrix: torch.Tensor = None) -> torch.Tensor:
        p = torch.tensor(0.0)
        for e in self.elems.values():
            p = p + e.potential_Energy(RGC, rotation_matrix=rotation_matrix)
        return p

    def structural_stiffness(self, RGC: torch.Tensor, rotation_matrix: torch.Tensor = None) -> list[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        K_values = []
        K_indices = []
        R_values = []
        R_indices = []
        for e in self.elems.values():
            Ra_indice, Ra_values, Ka_indice, Ka_value = e.structural_Force(
                RGC=RGC, rotation_matrix=rotation_matrix)
            K_values.append(Ka_value)
            K_indices.append(Ka_indice)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)

        K_indices = torch.cat(K_indices, dim=1)
        K_values = torch.cat(K_values, dim=0)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)
        return R_indices, R_values, K_indices, K_values
    
    def structural_force(self, RGC: torch.Tensor, rotation_matrix: torch.Tensor = None) -> list[torch.Tensor, torch.Tensor]:
        
        R_values = []
        R_indices = []
        for e in self.elems.values():
            Ra_indice, Ra_values = e.structural_Force(
                RGC=RGC, rotation_matrix=rotation_matrix, if_onlyforce=True)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)
        return R_indices, R_values

    # endregion

    # region dynamic

    def get_mass_matrix(self, rotation_matrix: torch.Tensor = None):
        M_indices = []
        M_values = []
        for e in self.elems.values():
            Me_indice, Me_value = e.get_mass_matrix(rotation_matrix=rotation_matrix)
            M_indices.append(Me_indice)
            M_values.append(Me_value)

        M_indices = torch.cat(M_indices, dim=1)
        M_values = torch.cat(M_values, dim=0)

        return M_indices, M_values

    # endregion


#========= Source code for Serializable.BaseObj.Instance =========#
class Instance(BaseObj):

    _serialized_attributes = ['part_name', '_translation', '_rotation', 'external_surface']
    def __init__(self, part_name: str, translation: torch.Tensor = None, rotation: torch.Tensor = None, external_surface: str = '') -> None:
        """
        Create an instance of a part.

        Parameters:
            part (Part): The part to be instantiated.
        """

        super().__init__()

        self.part_name = part_name

        self.part: Part = None

        if translation is not None:
            self._translation = translation
        else:
            self._translation: torch.Tensor = torch.zeros(3)
            """the translation vector of the instance"""

        if rotation is not None:
            self._rotation = rotation
        else:
            self._rotation: torch.Tensor = torch.randn(3) * 0.0
            """the rotation vector of the instance defined in exponential map"""

        self.external_surface: str = external_surface
        """the name of the external surface for visualization"""

        theta = torch.norm(self._rotation)
        if theta == 0:
            self.rotation_matrix = None
        else:
            r = self._rotation / theta
            r = r.view(3, 1)
            R = torch.cos(theta) * torch.eye(3) + (1 - torch.cos(theta)) * (r @ r.t()) + torch.sin(theta) * torch.tensor([[0, -r[2, 0], r[1, 0]], [r[2, 0], 0, -r[0, 0]], [-r[1, 0], r[0, 0], 0]])
            self.rotation_matrix = R


    @property
    def elems(self) -> dict[str, BaseElement]:
        return self.part.elems
    
    @property
    def nodes(self) -> torch.Tensor:
        return self._transform(self.part.nodes, self._rotation) + self._translation.unsqueeze(0)
    
    @property
    def surfaces(self) -> _Surfaces:
        return self.part.surfaces

    @property
    def set_nodes(self) -> dict[str, np.ndarray]:
        return self.part.set_nodes

    @staticmethod
    def _transform(vector0: torch.Tensor, rotation_vector: torch.Tensor = None):
        """
        Rotate a 3D vector by a rotation vector
        :param rotation_vector: rotation vector (3,)
        :param vector0: 3D vector (n, 3)
        :return: 3D vector (n, 3)
        """
        vector0 = vector0.view(-1, 3)
        if rotation_vector is None:
            return vector0
        
        theta = torch.norm(rotation_vector)
        if theta == 0:
            return vector0
        else:
            rotation_vector = rotation_vector / theta
            rotation_vector = rotation_vector.view(1, 3)
            vector1 = vector0 * torch.cos(theta) + torch.cross(
                rotation_vector, vector0, dim=1) * torch.sin(
                    theta) + rotation_vector * (rotation_vector * vector0).sum(
                        dim=1).unsqueeze(-1) * (1 - torch.cos(theta))
        return vector1

    def set_required_DoFs(self, RGC_remain_index):
        for e in self.part.elems.values():
            RGC_remain_index[self._RGC_index] = e.set_required_DoFs(RGC_remain_index[self._RGC_index])
        return RGC_remain_index
    
    def set_RGC_index(self, index):
        super().set_RGC_index(index)
    
    def initialize(self, assembly: Assembly):
        
        super().initialize(assembly=assembly)

    def get_mass_matrix(self):
        mass_indices, mass_values = self.part.get_mass_matrix(self.rotation_matrix)
        return mass_indices + self._index_start, mass_values

    def refine_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        RGC_out = RGC
        RGC_out[self._RGC_index] = self.part.refine_RGC(RGC[self._RGC_index])
        return RGC_out
    
    def potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return self.part.potential_energy(RGC=RGC[self._RGC_index], rotation_matrix=self.rotation_matrix)
    
    def structural_stiffness(self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs) -> list[torch.Tensor]:

        if if_onlyforce:
            R_indices, R_values = self.part.structural_force(RGC[self._RGC_index], rotation_matrix=self.rotation_matrix)
            return R_indices + self._index_start, R_values
        
        R_indices, R_values, K_indices, K_values = self.part.structural_stiffness(RGC[self._RGC_index], rotation_matrix=self.rotation_matrix)
        return R_indices + self._index_start, R_values, K_indices + self._index_start, K_values
    
    def extract_surfaces(self, name: str) -> list[BaseSurface]:
        surfaces = self.part.extract_surfaces(name)
        return surfaces

    def get_mesh(self, RGC: list[torch.Tensor] = None, surf_name: str = None) -> pv.PolyData:
        """
        Get the mesh of the instance for visualization.
        Args:
            surf_name (str, optional): Name of the surface set to visualize. 
                If None, use the external_surface attribute. Defaults to None.
        Returns:
            pv.PolyData: The mesh of the instance.
        """
        import pyvista as pv

        if surf_name is None:
            surf_name = self.external_surface
        if surf_name == '':
            return pv.PolyData()
        
        triangles = self.surfaces.get_trimesh(surf_name).cpu().numpy()
        nodes_transformed = self.nodes.cpu().numpy()

        if RGC is not None:
            nodes_transformed += RGC[self._RGC_index].cpu().numpy()

        mesh = pv.PolyData(nodes_transformed, np.hstack([ np.full((triangles.shape[0], 1), 3), triangles ]))
        return mesh
    
    def show(self, RGC: list[torch.Tensor] = None, surf_name: str = None):
        mesh = self.get_mesh(RGC=RGC, surf_name=surf_name)
        if mesh.n_points == 0:
            print("No surface to show.")
            return
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        plotter.show()


#========= Source code for Serializable.BaseObj.ReferencePoint =========#
class ReferencePoint(BaseObj):
    """
    ReferencePoints class for handling reference points in a finite element analysis (FEA) framework.
    This class is used to manage the coordinates of reference points, which can be used for various purposes such as boundary conditions, loads, or other constraints in the FEA model.
    """

    def __init__(self, node: list[float] | torch.Tensor = None) -> None:
        """
        Initialize the ReferencePoints class.

        Parameters:
        node (torch.Tensor): A tensor of shape (3) representing the coordinates of the reference point.
        """
        super().__init__()
        if isinstance(node, list):
            node = torch.tensor(node)
        elif isinstance(node, np.ndarray):
            node = torch.tensor(node.tolist())
        self.node = node
        self._RGC_requirements = 6


#========= Source code for Serializable.BaseObj.BaseLoad =========#
class BaseLoad(BaseObj):

    def __init__(self) -> None:
        super().__init__()
        self._indices_matrix: torch.Tensor = torch.zeros([2, 0],
                                                        dtype=torch.int)
        """
            the coo index of the stiffness matricx of structural stress
        """

        self._indices_force: torch.Tensor
        """
            the coo index of the tructural stress
        """

        self._index_matrix_coalesce: torch.Tensor = torch.zeros([0],
                                                            dtype=torch.int)
        """
            the start index of the stiffness matricx of structural stress
        """

        
        self._parameters: torch.Tensor = torch.zeros(0, dtype=torch.float64)
        """The parameters of this object.
        This is a 1D tensor containing all the parameters of this object.
        """

    def initialize(self, assembly: Assembly):
        super().initialize(assembly)
    
    def get_stiffness(self,
                RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the stiffness matrix and force vector for the self-contact load.

        Args:
            RGC (list[torch.Tensor]): The global coordinates of the nodes.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The stiffness matrix and force vector for the self-contact load.
                - F_indices: The indices of the force vector.
                - F_values: The values of the force vector.
                - K_indices: The indices of the stiffness matrix.
                - K_values: The values of the stiffness matrix.
        """

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """Get the potential energy for the self-contact load."""
        raise NotImplementedError("get_potential_energy method not implemented")
    
    @staticmethod
    def get_F0():
        raise NotImplementedError("get_F0 method not implemented")


#========= Source code for Serializable.BaseObj.BaseLoad.Concentrate_Force =========#
class Concentrate_Force(BaseLoad):

    def __init__(self, rp_name: str, force: list[float]) -> None:
        super().__init__()
        self.rp_name = rp_name
        self.rp_index: int = None
        self._parameters = torch.tensor(force, dtype=torch.float64)

    @property
    def force(self) -> torch.Tensor:
        return self._parameters
    
    @force.setter
    def force(self, value: list[float] | torch.Tensor) -> None:
        if isinstance(value, list):
            self._parameters = torch.tensor(value, dtype=torch.float64)
        else:
            self._parameters = value.to(torch.float64)

    def initialize(self, assembly):
        super().initialize(assembly)
        self.rp_index = assembly.get_reference_point(self.rp_name)._RGC_index
        self._indices_force = torch.arange(assembly.RGC_list_indexStart[self.rp_index], assembly.RGC_list_indexStart[self.rp_index]+3)


    def get_stiffness(self,
                RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if if_onlyforce:
            return self._indices_force, self.force
        
        return self._indices_force, self.force, torch.zeros([2, 0], dtype=torch.int), torch.zeros([0])

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        if type(self.force) == list:
            self.force = torch.tensor(self.force)
        return (self.force * RGC[self.rp_index][:3]).sum()

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self.rp_index][:3] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseLoad.Pressure =========#
class Pressure(BaseLoad):

    _serialized_attributes = ['surface_set', 'instance_name', '_parameters']

    def __init__(self, instance_name: str, surface_set: str, pressure: float) -> None:
        """
        initialize the pressure load on the surface element
        
        Args:
            surface_element (list[tuple[int, np.ndarray]]): the element index and the surface element index
            pressure (float): the pressure value
        """
        super().__init__()
        self.surface_set = surface_set
        self.instance_name = instance_name
        """Record the instance name and surface name
        """

        self.surface_element: list[BaseSurface]
        """
            the surface element
        """

        self._parameters = torch.tensor([pressure], dtype=torch.float64).flatten()
        """
        The pressure value applied to the surface element.
        """

        self._Vdot_indices: torch.Tensor
        """
        Indices for the force vector, used to apply the pressure load.
        """

        self._Vdot_2_indices: torch.Tensor
        """
        Indices for the second-order stiffness matrix, used to compute the stiffness contributions.
        """

        self._load_index: int

    @property
    def pressure(self) -> float:
        """
        Get the pressure value.
        """
        return self._parameters[0]
    
    @pressure.setter
    def pressure(self, value: float) -> None:
        """
        Set the pressure value.
        """
        self._parameters = torch.tensor([value], dtype=torch.float64).flatten()

    def initialize(self, assembly):
        super().initialize(assembly)

        self._load_index = self._assembly.get_instance(self.instance_name)._RGC_index
        index_offset = self._assembly.RGC_list_indexStart[self._load_index]

        self.surface_element = assembly.get_instance(self.instance_name).surfaces.get_elements(self.surface_set)

        _Vdot_indices = []
        _Vdot_2_indices = []

        for surf_ind in range(len(self.surface_element)):
            surf_elem = self.surface_element[surf_ind]
            if surf_elem._elems.shape[0] == 0:
                continue

            Vdot_indices = torch.stack([surf_elem._elems*3, surf_elem._elems*3+1, surf_elem._elems*3+2], dim=1).to(torch.int64)
            
            
            Vdot_2_indices = torch.stack([
                    surf_elem._elems.reshape([surf_elem._elems.shape[0], 1, surf_elem._elems.shape[1], 1, 1]).repeat([1, 3, 1, 3, surf_elem._elems.shape[1]]).flatten(),
                    torch.arange(3, device=surf_elem._elems.device).reshape([1, 3, 1, 1, 1]).repeat([surf_elem._elems.shape[0], 1, surf_elem._elems.shape[1], 3, surf_elem._elems.shape[1]]).flatten(),
                    surf_elem._elems.reshape([surf_elem._elems.shape[0], 1, 1, 1, surf_elem._elems.shape[1]]).repeat([1, 3, surf_elem._elems.shape[1], 3, 1]).flatten(),
                    torch.arange(3, device=surf_elem._elems.device).reshape([1, 1, 1, 3, 1]).repeat([surf_elem._elems.shape[0], 3, surf_elem._elems.shape[1], 1, surf_elem._elems.shape[1]]).flatten()
                ], dim=0).to(torch.int64)
            
            Vdot_2_indices = torch.stack([
                Vdot_2_indices[0] * 3 + Vdot_2_indices[1],
                Vdot_2_indices[2] * 3 + Vdot_2_indices[3]
            ], dim=0).to(torch.int64)

            _Vdot_indices.append(Vdot_indices)
            _Vdot_2_indices.append(Vdot_2_indices)
        
        self._Vdot_indices = torch.cat(_Vdot_indices, dim=0) + index_offset
        self._Vdot_2_indices = torch.cat(_Vdot_2_indices, dim=1) + index_offset
        
    def get_stiffness(self,
                RGC: list[torch.Tensor], if_onlyforce=False, *args, **kwargs):
        
        node_pos = self._assembly.get_instance(self.instance_name).nodes + RGC[self._load_index]

        # node_pos.requires_grad_()

        # V = self.get_potential_energy(RGC) / self.pressure * (-1)

        for surf_ind in range(len(self.surface_element)):
            surf_elem = self.surface_element[surf_ind]
            if surf_elem._elems.shape[0] == 0:
                continue
            
            # evaluate the deformed position and its derivatives at the Gaussian points
            shape_fun_added = torch.cat([surf_elem.shape_function_gaussian[0].unsqueeze(1),
                surf_elem.shape_function_gaussian[1]], dim=1)
            
            # V = [r, rdg, rdr] the mixed product of the deformed position and its derivatives
            # r_added_gaussian = [g, e, i, m] the deformed
            r_added_gaussian = torch.zeros([surf_elem._num_gaussian, surf_elem._elems.shape[0], 3, 3])
            
            for a in range(surf_elem.num_nodes_per_elem):
                r_added_gaussian += torch.einsum('gm, ei->geim', shape_fun_added[:, :, a],
                                           node_pos[surf_elem._elems[:, a]])
            
            

            det_ra = r_added_gaussian.det()
            inv_ra = r_added_gaussian.inverse()

            det_radra = torch.einsum('ge, gemi->geim', det_ra, inv_ra)

            Vdra = 1 / 3 * torch.einsum('g, geim->geim', surf_elem.gaussian_weight, det_radra)



            Vdot_values = torch.einsum('gma, geim->eia', shape_fun_added, Vdra)

            if if_onlyforce:
                continue

            part_I = torch.einsum('geim, genj->geimjn', det_radra, inv_ra)
            part_II = part_I.permute([0, 1, 2, 5, 4, 3])

            Vdra_2 = 1 / 3 * torch.einsum('g, geimjn->geimjn', surf_elem.gaussian_weight, part_I - part_II)
            Vdot_2_values = torch.einsum('gma, gnb, geimjn->eiajb', shape_fun_added, shape_fun_added, Vdra_2)
        
        if if_onlyforce:
            return self._Vdot_indices.flatten(), -self.pressure * Vdot_values.flatten()

        return self._Vdot_indices.flatten(), -self.pressure * Vdot_values.flatten(), self._Vdot_2_indices, -self.pressure * Vdot_2_values.flatten()

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:

        node_pos = self._assembly.get_instance(self.instance_name).nodes + RGC[self._load_index]

        V = torch.scalar_tensor(0)
        for surf_ind in range(len(self.surface_element)):
            surf_elem = self.surface_element[surf_ind]
            if surf_elem._elems.shape[0] == 0:
                continue
            # evaluate the deformed position and its derivatives at the Gaussian points
            shape_fun_added = torch.cat([surf_elem.shape_function_gaussian[0].unsqueeze(1),
                surf_elem.shape_function_gaussian[1]], dim=1)
            
            # V = [r, rdg, rdr] the mixed product of the deformed position and its derivatives
            # r_added_gaussian = [g, e, i, m] the deformed
            r_added_gaussian = torch.zeros([surf_elem._num_gaussian, surf_elem._elems.shape[0], 3, 3])
            
            for a in range(surf_elem.num_nodes_per_elem):
                r_added_gaussian += torch.einsum('gm, ei->geim', shape_fun_added[:, :, a],
                                           node_pos[surf_elem._elems[:, a]])
            
            # evaluate the volume of the closed shell
            V_now = surf_elem.gaussian_weight.view([-1, 1]) * r_added_gaussian.det() 
            V += V_now.sum() / 3.

        return -self.pressure * V

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        for surf_ind in range(len(self.surface_element)):

            RGC_remain_index[self._load_index][self.surface_element[surf_ind]._elems.flatten().unique().cpu()] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseLoad.Moment =========#
class Moment(BaseLoad):

    def __init__(self, rp_name: str, moment: list[float]) -> None:
        super().__init__()
        self.rp_name = rp_name
        self.rp_index: int = None
        self._parameters = torch.tensor(moment, dtype=torch.float64)

    @property
    def moment(self) -> torch.Tensor:
        return self._parameters
    
    @moment.setter
    def moment(self, value: list[float] | torch.Tensor) -> None:
        if isinstance(value, list):
            self._parameters = torch.tensor(value, dtype=torch.float64)
        else:
            self._parameters = value.to(torch.float64)

    def initialize(self, assembly):
        super().initialize(assembly)
        self.rp_index = assembly.get_reference_point(self.rp_name)._RGC_index
        self._indices_force = torch.arange(assembly.RGC_list_indexStart[self.rp_index]+3, assembly.RGC_list_indexStart[self.rp_index]+6)

    def get_stiffness(self,
                RGC: list[torch.Tensor], if_onlyforce=False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:

        if if_onlyforce:
            return self._indices_force, self.moment

        return self._indices_force, self.moment, torch.zeros([2, 0], dtype=torch.int), torch.zeros([0])

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        if type(self.moment) == list:
            self.moment = torch.tensor(self.moment)
        return (self.moment * RGC[self.rp_index][3:]).sum()

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self.rp_index][3:] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseLoad.ContactBase =========#
class ContactBase(BaseLoad):



    def __init__(self,
                 penalty_distance_f: float = 1e-5,
                 penalty_factor_f: float = 40.0,
                 penalty_start_g: float = -0.8,
                 penalty_end_g: float = -0.85,
                 penalty_threshold_h: float = 1.5,
                 penalty_ratio_h: float = 0.9,
                 mesh_size: float = 1.0):
        """
        Initialize the base contact load with common parameters.

        Args:
            penalty_distance_f (float): The penalty distance for contact. When the distance between nodes is less than this value, a penalty is applied.
            penalty_factor_f (float): The penalty factor f for contact.
            penalty_start_g (float): The penalty degree for the angle factor g. The degree of the penalty function.
            penalty_end_g (float): The penalty threshold for the angle factor g.
            penalty_threshold_h (float): The penalty threshold for contact.
            penalty_ratio_h (float): The penalty ratio for contact.
            mesh_size (float): The size of the mesh elements.
        """
        super().__init__()

        self._parameters = torch.tensor([
            penalty_distance_f,
            penalty_factor_f,
            penalty_start_g,
            penalty_end_g,
            penalty_threshold_h,
            penalty_ratio_h,
        ], dtype=torch.float64)

        self._point_pairs: torch.Tensor
        """The point pairs that need to be considered for self-contact."""

        self.is_self_contact: bool
        """Whether this is self-contact (True) or two-surface contact (False)."""

        self.surface_element1: BaseSurface
        """The first surface element for contact."""

        self.surface_element2: BaseSurface
        """The second surface element for contact."""

        self.surface_name1: str
        """The name of the first surface to apply the load on."""

        self.surface_name2: str
        """The name of the second surface to apply the load on."""

        self.instance_name1: str
        """The name of the first instance to apply the load on."""

        self.instance_name2: str
        """The name of the second instance to apply the load on."""

        self.mesh_size = mesh_size
        """The size of the mesh elements, used for filtering point pairs in contact detection."""


    @property
    def penalty_distance_f(self) -> float:
        return self._parameters[0]
    @penalty_distance_f.setter
    def penalty_distance_f(self, value: float):
        self._parameters[0] = value

    @property
    def penalty_factor_f(self) -> float:
        return self._parameters[1]
    @penalty_factor_f.setter
    def penalty_factor_f(self, value: float):
        self._parameters[1] = value
        
    @property
    def penalty_start_g(self) -> float:
        return self._parameters[2]
    @penalty_start_g.setter
    def penalty_start_g(self, value: float):
        self._parameters[2] = value

    @property
    def penalty_end_g(self) -> float:
        return self._parameters[3]
    @penalty_end_g.setter
    def penalty_end_g(self, value: float):
        self._parameters[3] = value

    @property
    def penalty_threshold_h(self) -> float:
        return self._parameters[4]
    @penalty_threshold_h.setter
    def penalty_threshold_h(self, value: float):
        self._parameters[4] = value

    @property
    def penalty_ratio_h(self) -> float:
        return self._parameters[5]
    @penalty_ratio_h.setter
    def penalty_ratio_h(self, value: float):
        self._parameters[5] = value


    def _filter_point_pairs(self, surface_element1: BaseSurface, surface_element2: BaseSurface, nodes1: torch.Tensor, nodes2: torch.Tensor, max_search_length_ratio: float = 1.5):
        """
        Filter point pairs between surfaces for contact detection.
        
        Args:
            surface_element1: First surface element
            surface_element2: Second surface element (same as first for self-contact)
            nodes: Node positions
            is_self_contact: Whether this is self-contact (affects diagonal filtering)
            
        Returns:
            tuple: (point_pairs, ratio_d) for contact detection
        """

        # Get Gaussian points for both surfaces
        elems_gaussian1 = surface_element1.gaussian_points_position(nodes1)
        elems_gaussian2 = surface_element2.gaussian_points_position(nodes2)
        
        # Calculate midpoints for initial distance filtering
        elems_mid1 = elems_gaussian1.mean(dim=0).cpu()
        elems_mid2 = elems_gaussian2.mean(dim=0).cpu()
        
        # Calculate distances between surface midpoints
        if not self.is_self_contact:
            points = torch.cat([elems_mid1, elems_mid2], dim=0).detach().cpu().numpy()
        else:
            points = elems_mid1.detach().cpu().numpy()
        kdtree = scipy.spatial.cKDTree(points)
        pairs = torch.from_numpy(kdtree.query_pairs(max_search_length_ratio * self.penalty_threshold_h + self.mesh_size, output_type='ndarray')).to(nodes1.device).T
        index_revert = torch.where(pairs[0] >= pairs[1])[0]
        pairs[:, index_revert] = pairs[:, index_revert][[1, 0]]
        if not self.is_self_contact:
            pairs = pairs[:, pairs[0] < elems_mid1.shape[0]]
            pairs = pairs[:, pairs[1] >= elems_mid1.shape[0]]
            pairs[1] -= elems_mid1.shape[0]
        
        self._point_pairs = pairs


#========= Source code for Serializable.BaseObj.BaseLoad.ContactBase.ContactSelf =========#
class ContactSelf(ContactBase):
    """
    Class representing self-contact loads in the finite element model.
    """

    _serialized_attributes = ['surface_name', 
                              'instance_name', 
                              '_ignore_min_normal', 
                              '_ignore_max_normal', 
                              '_initial_detact_ratio', 
                              '_parameters']

    def __init__(self, instance_name: str, surface_name: str,
                 ignore_min_normal: float = -0.5,
                 ignore_max_normal: float = 1.5, 
                 initial_detact_ratio: float = 1.5, 
                 penalty_distance_f: float = 1e-5,
                 penalty_factor_f: float = 40.0,
                 penalty_start_g: float = -0.8,
                 penalty_end_g: float = -0.85,
                 penalty_threshold_h: float = 1.5,
                 penalty_ratio_h: float = 0.9,
                 mesh_size: float = 1.0):
        """
        Initialize the self-contact load.

        Args:
            surface_name (str): The name of the surface to apply the load on.
            **kwargs: Additional parameters passed to ContactBase.
        """

        super().__init__(
            penalty_distance_f=penalty_distance_f,
            penalty_factor_f=penalty_factor_f,
            penalty_start_g=penalty_start_g,
            penalty_end_g=penalty_end_g,
            penalty_threshold_h=penalty_threshold_h,
            penalty_ratio_h=penalty_ratio_h,
            mesh_size=mesh_size)

        self._ignore_min_normal = ignore_min_normal
        """The minimum initial normal distance to ignore for contact."""
        self._ignore_max_normal = ignore_max_normal
        """The maximum initial normal distance to ignore for contact."""

        self.surface_name = surface_name
        """The name of the surface to apply the load on."""

        self.instance_name = instance_name
        """The name of the instance to apply the load on."""

        self.surface_name1 = self.surface_name2 = surface_name
        self.instance_name1 = self.instance_name2 = instance_name

        self.surface_element: BaseSurface
        """The list of surface elements for self-contact."""

        self.is_self_contact = True

        self._initial_detact_ratio = initial_detact_ratio
        """The initial detach ratio to avoid the initial intersection of surfaces."""
    
    def initialize(self, assembly):
        
        super().initialize(assembly)

        # filter the point pairs
        self.surface_element = assembly.get_instance(self.instance_name).surfaces.get_elements(self.surface_name)[0]

        self.surface_element1 = self.surface_element2 = self.surface_element

        instance = assembly.get_instance(self.instance_name)

        self._filter_point_pairs(
            self.surface_element, self.surface_element, instance.nodes)

    def _filter_point_pairs(self, surface_element1: BaseSurface, surface_element2: BaseSurface, nodes: torch.Tensor):
        super()._filter_point_pairs(surface_element1, surface_element2, nodes1=nodes, nodes2=nodes, max_search_length_ratio=self._initial_detact_ratio)
        
        def _ratio_d_func(dx: torch.Tensor, dm: torch.Tensor):
            """
            Calculate the ratio for self-contact to avoid the calculation of the nearest distance.

            Args:
                dx (torch.Tensor): The normalized distance vector between points.
                dm (torch.Tensor): The difference in normal vectors between points.

            Returns:
                torch.Tensor: The ratio for self-contact.
            """
            dx = dx / dx.norm(dim=-1, keepdim=True)

            T = - (dm * dx).sum(-1)
            T = (T - self._ignore_min_normal) / (self._ignore_max_normal - self._ignore_min_normal)
            T = T.clamp(0, 1)
            return 6 * T**5 - 15 * T**4 + 10 * T**3

        instance = self._assembly.get_instance(self.instance_name)

        # Get surface normals
        normal1 = surface_element1.get_gaussian_normal(instance.nodes)
        normal2 = surface_element2.get_gaussian_normal(instance.nodes)
        elems_gaussian1 = surface_element1.gaussian_points_position(instance.nodes)
        elems_gaussian2 = surface_element2.gaussian_points_position(instance.nodes)
        normal1 = normal1 / normal1.norm(dim=-1, keepdim=True)
        normal2 = normal2 / normal2.norm(dim=-1, keepdim=True)

        # Calculate normal differences and position differences
        dm = normal1[:, None, self._point_pairs[0], :] - normal2[None, :, self._point_pairs[1], :]
        dy = elems_gaussian1[:, None, self._point_pairs[0], :] - elems_gaussian2[None, :, self._point_pairs[1], :]
        dr = dy / dy.norm(dim=-1, keepdim=True)

        # Calculate ratio based on normal alignment
        ratio_d = _ratio_d_func(dx=dr, dm=dm)
        index_remain = (ratio_d.sum([0, 1]) > 0)

        self._point_pairs = self._point_pairs[:, index_remain]
        self._ratio = ratio_d[:, :, index_remain] / ratio_d[:, :, index_remain]

    def get_potential_energy(self, RGC):

        instance = self._assembly.get_instance(self.instance_name)
        self._filter_point_pairs(self.surface_element, self.surface_element, instance.nodes + RGC[instance._RGC_index])

        weight = torch.einsum('gp, g, Gp, G->gGp', 
                              self.surface_element1.det_Jacobian[:, self._point_pairs[0]], 
                              self.surface_element1.gaussian_weight,
                              self.surface_element2.det_Jacobian[:, self._point_pairs[1]],
                              self.surface_element2.gaussian_weight)

        U = RGC[instance._RGC_index]
        # U = U.detach().clone().requires_grad_()
        Y = instance.nodes + U

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
        MM = (self.penalty_start_g - M) / (self.penalty_start_g-self.penalty_end_g)
        MM = MM.clamp(0, 1)
        f = MM**3 * (6*MM**2 - 15*MM + 10)

        D = (dn * dy).sum(dim=-1) / 2
        g = torch.exp(D * self.penalty_factor_f) * self.penalty_distance_f


        L = dy.norm(dim=-1)
        T = (self.penalty_threshold_h - L) / (self.penalty_ratio_h * self.penalty_threshold_h)
        T = T.clamp(0, 1)
        h = T**3 * (6*T**2 - 15*T + 10)

        penalty =g * f * h * weight
        potential_energy = penalty.sum()
        return -potential_energy


    def get_stiffness(self, RGC, if_onlyforce: bool = False, *args, **kwargs):
        

        instance = self._assembly.get_instance(self.instance_name)
        self._filter_point_pairs(self.surface_element, self.surface_element, instance.nodes + RGC[instance._RGC_index])

        weight0 = torch.einsum('gp, g, Gp, G->gGp', 
                              self.surface_element1.det_Jacobian[:, self._point_pairs[0]], 
                              self.surface_element1.gaussian_weight,
                              self.surface_element2.det_Jacobian[:, self._point_pairs[1]],
                              self.surface_element2.gaussian_weight)

        U = RGC[instance._RGC_index]
        # U = U.detach().clone().requires_grad_()
        Y = instance.nodes + U

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
        E0 = torch.zeros([num_g, num_p, 2, 2, 3], device=U.device) # e1/e2, y/n, 0/1/2

        E0[:, :, 0, 0] = y[:, self._point_pairs[0]]
        E0[:, :, 1, 0] = y[:, self._point_pairs[1]]
        E0[:, :, 0, 1] = n[:, self._point_pairs[0]]
        E0[:, :, 1, 1] = n[:, self._point_pairs[1]]

        dy0 = E0[:, None, :, 0, 0, :] - E0[None, :, :, 1, 0, :]
        dn0 = E0[:, None, :, 0, 1, :] - E0[None, :, :, 1, 1, :]

        M0 = (E0[:, None, :, 0, 1, :] * E0[None, :, :, 1, 1, :]).sum(dim=-1)
        MM0 = (self.penalty_start_g - M0) / (self.penalty_start_g-self.penalty_end_g)
        MM0 = MM0.clamp(0, 1)
        f0 = MM0**3 * (6*MM0**2 - 15*MM0 + 10)

        D0 = (dn0 * dy0).sum(dim=-1) / 2
        g0 = torch.exp(D0 * self.penalty_factor_f) * self.penalty_distance_f


        L0 = dy0.norm(dim=-1)
        T0 = (self.penalty_threshold_h - L0) / (self.penalty_ratio_h * self.penalty_threshold_h)
        T0 = T0.clamp(0, 1)
        h0 = T0**3 * (6*T0**2 - 15*T0 + 10)

        penalty = g0 * f0 * h0 * weight0

        # filter the zero penalty pairs
        index_remain_total = torch.where(penalty.sum([0,1])>1e-12)[0]

        
        num_p = index_remain_total.shape[0]

        if num_p == 0:
            # No active contact pairs
            if if_onlyforce:
                return torch.tensor([], dtype=torch.int64), torch.tensor([])
            return torch.tensor([], dtype=torch.int64), torch.tensor([]), torch.tensor([[], []], dtype=torch.int64), torch.tensor([])
        

        pdU_indices_total = [] 
        pdU_values_total = []
        pdU_2_indices_total = []
        pdU_2_values_total = []

        index_now = 0
        batch_size = 5000
        while True:

            index_remain = index_remain_total[index_now:index_now+batch_size]
            num_p = index_remain.shape[0]
            if index_remain.shape[0] == 0:
                break

            point_pairs = self._point_pairs[:, index_remain]
            D = D0[:, :, index_remain]
            M = M0[:, :, index_remain]
            MM = MM0[:, :, index_remain]
            E = E0[:, index_remain]
            T = T0[:, :, index_remain]
            L = L0[:, :, index_remain]
            dy = dy0[:, :, index_remain]
            dn = dn0[:, :, index_remain]
            f = f0[:, :, index_remain]
            g = g0[:, :, index_remain]
            h = h0[:, :, index_remain]
            weight = weight0[:, :, index_remain]
            
            # if index_remain.shape[0] > 0:
                # print('  Contact pairs: ', index_remain.shape[0], '\t surface name: ', self.surface_name)
                # from mayavi import mlab
                # ind = 0
                # point_pairs_show = point_pairs[:, [ind]]
                # mlab.figure()
                # mlab.triangular_mesh((self._fea.nodes+RGC[0]).cpu()[:, 0], (self._fea.nodes+RGC[0]).cpu()[:, 1], (self._fea.nodes+RGC[0]).cpu()[:, 2], self.surface_element._elems.cpu().numpy(), color=(0.5, 0.5, 0.5))
                # mlab.points3d(y[0][point_pairs_show[0], 0].cpu(), y[0][point_pairs_show[0], 1].cpu(), y[0][point_pairs_show[0], 2].cpu(), color=(1, 0, 0), scale_factor = 0.2)
                # mlab.points3d(y[0][point_pairs_show[1], 0].cpu(), y[0][point_pairs_show[1], 1].cpu(), y[0][point_pairs_show[1], 2].cpu(), color=(0, 0, 1), scale_factor = 0.2)

                # mlab.quiver3d(y[0][point_pairs_show[0], 0].cpu(), y[0][point_pairs_show[0], 1].cpu(), y[0][point_pairs_show[0], 2].cpu(), n[0][point_pairs_show[0], 0].cpu(), n[0][point_pairs_show[0], 1].cpu(), n[0][point_pairs_show[0], 2].cpu(), scale_factor=10.)
                # mlab.quiver3d(y[0][point_pairs_show[1], 0].cpu(), y[0][point_pairs_show[1], 1].cpu(), y[0][point_pairs_show[1], 2].cpu(), n[0][point_pairs_show[1], 0].cpu(), n[0][point_pairs_show[1], 1].cpu(), n[0][point_pairs_show[1], 2].cpu(), scale_factor=10.)
                # mlab.show()

            # # Compute the potential energy
            # potential_energy = penalty.sum()

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

            # g = torch.exp(D * self.penalty_factor_f) * self.penalty_distance_f
            gdD = (self.penalty_factor_f) * g
            gdD_2 = (self.penalty_factor_f**2) * g

            gdE = torch.zeros([num_g, num_g, num_p, 2, 2, 3])
            gdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', gdD / 2, dn)
            gdE[:, :, :, 1, 0, :] = -gdE[:, :, :, 0, 0, :]
            gdE[:, :, :, 0, 1, :] = torch.einsum('gGp, gGpi->gGpi', gdD / 2, dy)
            gdE[:, :, :, 1, 1, :] = -gdE[:, :, :, 0, 1, :]

            gdE_2 = torch.zeros([num_g, num_g, num_p, 2, 2, 3, 2, 2, 3])
            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dn, dn)
            gdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
            gdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dn, dy) + \
                    torch.einsum('gGp, ij->gGpij', gdD / 2, torch.eye(3))
            gdE_2[:, :, :, 0, 0, :, 0, 1, :] = tmp
            gdE_2[:, :, :, 0, 0, :, 1, 1, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 0, 1, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 1, 1, :] = tmp

            tmp = tmp.permute([0, 1, 2, 4, 3])
            gdE_2[:, :, :, 0, 1, :, 0, 0, :] = tmp
            gdE_2[:, :, :, 0, 1, :, 1, 0, :] = -tmp
            gdE_2[:, :, :, 1, 1, :, 0, 0, :] = -tmp
            gdE_2[:, :, :, 1, 1, :, 1, 0, :] = tmp

            temp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dy, dy)
            gdE_2[:, :, :, 0, 1, :, 0, 1, :] = temp
            gdE_2[:, :, :, 1, 1, :, 0, 1, :] = -temp
            gdE_2[:, :, :, 0, 1, :, 1, 1, :] = -temp
            gdE_2[:, :, :, 1, 1, :, 1, 1, :] = temp

            # MM = (self.penalty_start_g - M) / (self.penalty_start_g-self.penalty_end_g)
            # MM = MM.clamp(0, 1)
            # f = MM**3 * (6*MM**2 - 15*MM + 10)
            fdM = -30*MM**2*(MM-1)**2 / (self.penalty_start_g-self.penalty_end_g)
            fdM_2 = 60*MM*(MM-1)*(2*MM-1) / (self.penalty_start_g-self.penalty_end_g)**2
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
            hdL = -30*T**2*(T-1)**2 / (self.penalty_ratio_h * self.penalty_threshold_h)
            hdL_2 = 60*T*(T-1)*(2*T-1) / (self.penalty_ratio_h * self.penalty_threshold_h)**2
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

            pdE = pdE * weight[:, :, :, None, None, None]

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
            
            pdE_2 = torch.einsum('gGpmxinyj, gGp->gGpmxinyj', pdE_2, weight)
    
            # pdUe = torch.zeros([num_e, num_n, 3])
            pdEsum0 = pdE.sum(0)
            pdEsum1 = pdE.sum(1)
            pdUe_values0 = torch.einsum('gpxi, gpxial->pal', pdEsum1[:, :, 0], edUe[:, point_pairs[0]])
            pdUe_values1 = torch.einsum('Gpxi, Gpxial->pal', pdEsum0[:, :, 1], edUe[:, point_pairs[1]])

            # for i in range(point_pairs.shape[1]):
            #     pdUe[point_pairs[0, i]] += pdUe_values0[i]
            #     pdUe[point_pairs[1, i]] += pdUe_values1[i]

            pdU_values = torch.stack([pdUe_values0, pdUe_values1], dim=0)
            tri_ind = point_pairs

            pdU_indices = self.surface_element._elems[tri_ind].to(torch.int64)
            pdU_indices = torch.stack([pdU_indices*3, pdU_indices*3+1, pdU_indices*3+2], dim=-1)
            pdU_indices = pdU_indices.to(torch.get_default_device())

            # pdU = torch.zeros_like(Y).flatten().scatter_add_(0, pdU_indices.flatten(), pdU_values.flatten()).reshape([-1, 3])

            pdUe_2_values00 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(1)[:, :, 0, :, :, 0], edUe[:, point_pairs[0]], edUe[:, point_pairs[0]]) + \
                                torch.einsum('gpxi, gpxialbL->palbL', pdEsum1[:, :, 0], edUe_2[:, point_pairs[0]])
            
            pdUe_2_values01 = torch.einsum('gGpxiyj, gpxial, GpyjbL->palbL', pdE_2[:, :, :, 0, :, :, 1], edUe[:, point_pairs[0]], edUe[:, point_pairs[1]])

            pdUe_2_values10 = torch.einsum('gGpxiyj, Gpxial, gpyjbL->palbL', pdE_2[:, :, :, 1, :, :, 0], edUe[:, point_pairs[1]], edUe[:, point_pairs[0]])
            
            pdUe_2_values11 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(0)[:, :, 1, :, :, 1], edUe[:, point_pairs[1]], edUe[:, point_pairs[1]]) + \
                                torch.einsum('gpxi, gpxialbL->palbL', pdEsum0[:, :, 1], edUe_2[:, point_pairs[1]])

            pdU_2_values = torch.stack([pdUe_2_values00, pdUe_2_values01, pdUe_2_values10, pdUe_2_values11], dim=0)


            # pdU_2 = torch.sparse_coo_tensor(pdU_2_indices_.reshape([4, -1]), pdU_2_values.flatten(), size=Y.shape*2)


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
            pdU_2_indices = torch.stack([pdU_2_indices_[0]*3+pdU_2_indices_[1], pdU_2_indices_[2]*3+pdU_2_indices_[3]], dim=0).to(torch.get_default_device())

            pdU_indices_total.append(pdU_indices.flatten())
            pdU_values_total.append(pdU_values.flatten())
            pdU_2_indices_total.append(pdU_2_indices.reshape([2, -1]))
            pdU_2_values_total.append(pdU_2_values.flatten())

            index_now += batch_size
        
        
        index_start = self._assembly.RGC_list_indexStart[instance._RGC_index]
        pdU_indices = torch.cat(pdU_indices_total, dim=0)
        pdU_values = torch.cat(pdU_values_total, dim=0)

        if if_onlyforce:
            return pdU_indices + index_start, -pdU_values

        pdU_2_indices = torch.cat(pdU_2_indices_total, dim=1)
        pdU_2_values = torch.cat(pdU_2_values_total, dim=0)

        
        return pdU_indices + index_start, -pdU_values, pdU_2_indices + index_start, -pdU_2_values

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self._assembly.get_instance(self.instance_name)._RGC_index][self.surface_element._elems.flatten().unique().cpu()] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseLoad.ContactBase.Contact =========#
class Contact(ContactBase):
    """
    Contact between two surfaces.

    Args:
        surface_name1 (str): The name of the first surface element.
        surface_name2 (str): The name of the second surface element.
        **kwargs: Additional parameters passed to ContactBase.
    """

    _serialized_attributes = ['instance_name1', 'instance_name2', 'surface_name1', 'surface_name2', '_parameters', 'mesh_size']

    def __init__(self,
            instance_name1: str,
            instance_name2: str,
            surface_name1: str,
            surface_name2: str,
            penalty_distance_f: float = 1e-5,
            penalty_factor_f: float = 40.0,
            penalty_start_g: float = -0.8,
            penalty_end_g: float = -0.85,
            penalty_threshold_h: float = 1.5,
            penalty_ratio_h: float = 0.9,
            mesh_size: float = 1.0):
        
        super().__init__(
            penalty_distance_f=penalty_distance_f,
            penalty_factor_f=penalty_factor_f,
            penalty_start_g=penalty_start_g,
            penalty_end_g=penalty_end_g,
            penalty_threshold_h=penalty_threshold_h,
            penalty_ratio_h=penalty_ratio_h,
            mesh_size=mesh_size)

        self.surface_name1 = surface_name1
        """The name of the first surface to apply the load on."""

        self.surface_name2 = surface_name2
        """The name of the second surface to apply the load on."""

        self.surface_element1: BaseSurface
        """The first surface element for contact."""

        self.surface_element2: BaseSurface
        """The second surface element for contact."""

        self.instance_name1 = instance_name1
        """The name of the first instance containing the surface."""

        self.instance_name2 = instance_name2
        """The name of the second instance containing the surface."""

        self.is_self_contact = False

    def initialize(self, assembly):
        super().initialize(assembly)

        # Get surface elements from FEA model
        self.surface_element1 = assembly.get_instance(self.instance_name1).surfaces.get_elements(self.surface_name1)[0]
        self.surface_element2 = assembly.get_instance(self.instance_name2).surfaces.get_elements(self.surface_name2)[0]

        # Filter point pairs between the two surfaces
        self._filter_point_pairs(
            self.surface_element1, self.surface_element2, assembly.get_instance(self.instance_name1).nodes, assembly.get_instance(self.instance_name2).nodes)
        
    def get_potential_energy(self, RGC):
        
        instance1 = self._assembly.get_instance(self.instance_name1)
        instance2 = self._assembly.get_instance(self.instance_name2)
        self._filter_point_pairs(self.surface_element1, self.surface_element2, 
                                 instance1.nodes + RGC[instance1._RGC_index], 
                                 instance2.nodes + RGC[instance2._RGC_index])

        weight = torch.einsum('ge, g, Ge, G->gGe', 
                              self.surface_element1.det_Jacobian[:, self._point_pairs[0]], 
                              self.surface_element1.gaussian_weight,
                              self.surface_element2.det_Jacobian[:, self._point_pairs[1]],
                              self.surface_element2.gaussian_weight)

        # U = U.clone().detach().requires_grad_(True)
        Y1 = instance1.nodes + RGC[instance1._RGC_index]
        Y2 = instance2.nodes + RGC[instance2._RGC_index]

        num_g1 = self.surface_element1._num_gaussian
        num_g2 = self.surface_element2._num_gaussian
        num_e1 = self.surface_element1._elems.shape[0]
        num_e2 = self.surface_element2._elems.shape[0]
        num_n1 = self.surface_element1.num_nodes_per_elem
        num_n2 = self.surface_element2.num_nodes_per_elem

        # Calculate positions and normals for both surfaces
        Ye1 = Y1[self.surface_element1._elems]
        Ye2 = Y2[self.surface_element2._elems]

        y1 = torch.einsum('eai, ga->gei', Ye1, self.surface_element1.shape_function_gaussian[0])
        y2 = torch.einsum('eai, ga->gei', Ye2, self.surface_element2.shape_function_gaussian[0])

        NR1 = torch.einsum('gma, eai->gemi', self.surface_element1.shape_function_gaussian[1], Ye1)
        NR2 = torch.einsum('gma, eai->gemi', self.surface_element2.shape_function_gaussian[1], Ye2)
        
        N1 = torch.cross(NR1[:, :, 0, :], NR1[:, :, 1, :], dim=-1)
        N2 = torch.cross(NR2[:, :, 0, :], NR2[:, :, 1, :], dim=-1)

        nnorm1 = N1.norm(dim=-1)
        nnorm2 = N2.norm(dim=-1)
        n1 = N1 / nnorm1[:, :, None]
        n2 = N2 / nnorm2[:, :, None]

        num_p = self._point_pairs.shape[1]
        
        # Create extended tensor for two surfaces
        E1 = torch.zeros([num_g1, num_p, 2, 3], device=Y1.device)
        E1[:, :, 0] = y1[:, self._point_pairs[0]]
        E1[:, :, 1] = n1[:, self._point_pairs[0]]

        E2 = torch.zeros([num_g2, num_p, 2, 3], device=Y2.device)
        E2[:, :, 0] = y2[:, self._point_pairs[1]]
        E2[:, :, 1] = n2[:, self._point_pairs[1]]

        dy = E1[:, None, :, 0, :] - E2[None, :, :, 0, :]
        dn = E1[:, None, :, 1, :] - E2[None, :, :, 1, :]

        M = (E1[:, None, :, 1, :] * E2[None, :, :, 1, :]).sum(dim=-1)
        MM = (self.penalty_start_g - M) / (self.penalty_start_g - self.penalty_end_g)
        MM = MM.clamp(0, 1)
        f = MM**3 * (6*MM**2 - 15*MM + 10)

        D = (dn * dy).sum(dim=-1) / 2
        g = torch.exp(D * self.penalty_factor_f) * self.penalty_distance_f
        
        L = dy.norm(dim=-1)
        T = (self.penalty_threshold_h - L) / (self.penalty_ratio_h * self.penalty_threshold_h)
        T = T.clamp(0, 1)
        h = T**3 * (6*T**2 - 15*T + 10)

        penalty = g * f * h * weight
        
        # Compute the potential energy
        potential_energy = penalty.sum()
        return -potential_energy

    def get_stiffness(self, RGC, if_onlyforce=False, *args, **kwargs):
        instance1 = self._assembly.get_instance(self.instance_name1)
        instance2 = self._assembly.get_instance(self.instance_name2)
        self._filter_point_pairs(self.surface_element1, self.surface_element2, 
                                 instance1.nodes + RGC[instance1._RGC_index], 
                                 instance2.nodes + RGC[instance2._RGC_index])

        weight0 = torch.einsum('gp, g, Gp, G->gGp', 
                              self.surface_element1.det_Jacobian[:, self._point_pairs[0]], 
                              self.surface_element1.gaussian_weight,
                              self.surface_element2.det_Jacobian[:, self._point_pairs[1]],
                              self.surface_element2.gaussian_weight)

        # U = U.clone().detach().requires_grad_(True)
        Y1 = instance1.nodes + RGC[instance1._RGC_index]
        Y2 = instance2.nodes + RGC[instance2._RGC_index]

        num_g1 = self.surface_element1._num_gaussian
        num_g2 = self.surface_element2._num_gaussian
        num_e1 = self.surface_element1._elems.shape[0]
        num_e2 = self.surface_element2._elems.shape[0]
        num_n1 = self.surface_element1.num_nodes_per_elem
        num_n2 = self.surface_element2.num_nodes_per_elem

        # Calculate positions and normals for both surfaces
        Ye1 = Y1[self.surface_element1._elems]
        Ye2 = Y2[self.surface_element2._elems]

        y1 = torch.einsum('eai, ga->gei', Ye1, self.surface_element1.shape_function_gaussian[0])
        y2 = torch.einsum('eai, ga->gei', Ye2, self.surface_element2.shape_function_gaussian[0])

        NR1 = torch.einsum('gma, eai->gemi', self.surface_element1.shape_function_gaussian[1], Ye1)
        NR2 = torch.einsum('gma, eai->gemi', self.surface_element2.shape_function_gaussian[1], Ye2)
        
        N1 = torch.cross(NR1[:, :, 0, :], NR1[:, :, 1, :], dim=-1)
        N2 = torch.cross(NR2[:, :, 0, :], NR2[:, :, 1, :], dim=-1)

        nnorm1 = N1.norm(dim=-1)
        nnorm2 = N2.norm(dim=-1)
        n1 = N1 / nnorm1[:, :, None]
        n2 = N2 / nnorm2[:, :, None]

        num_p = self._point_pairs.shape[1]
        
        # Create extended tensor for two surfaces
        E10 = torch.zeros([num_g1, num_p, 2, 3], device=Y1.device)
        E10[:, :, 0] = y1[:, self._point_pairs[0]]
        E10[:, :, 1] = n1[:, self._point_pairs[0]]

        E20 = torch.zeros([num_g2, num_p, 2, 3], device=Y2.device)
        E20[:, :, 0] = y2[:, self._point_pairs[1]]
        E20[:, :, 1] = n2[:, self._point_pairs[1]]
        dy0 = E10[:, None, :, 0, :] - E20[None, :, :, 0, :]
        dn0 = E10[:, None, :, 1, :] - E20[None, :, :, 1, :]

        M0 = (E10[:, None, :, 1, :] * E20[None, :, :, 1, :]).sum(dim=-1)
        MM0 = (self.penalty_start_g - M0) / (self.penalty_start_g - self.penalty_end_g)
        MM0 = MM0.clamp(0, 1)
        f0 = MM0**3 * (6*MM0**2 - 15*MM0 + 10)

        D0 = (dn0 * dy0).sum(dim=-1) / 2
        g0 = torch.exp(D0 * self.penalty_factor_f) * self.penalty_distance_f
        
        L0 = dy0.norm(dim=-1)
        T0 = (self.penalty_threshold_h - L0) / (self.penalty_ratio_h * self.penalty_threshold_h)
        T0 = T0.clamp(0, 1)
        h0 = T0**3 * (6*T0**2 - 15*T0 + 10)

        penalty = g0 * f0 * h0 * weight0

        # Filter zero penalty pairs
        index_remain_total = torch.where(penalty.sum([0,1]) > 0)[0]

        if index_remain_total.shape[0] == 0:
            # No active contact pairs
            if if_onlyforce:
                return torch.tensor([], dtype=torch.int64), torch.tensor([])
            return torch.tensor([], dtype=torch.int64), torch.tensor([]), torch.tensor([[], []], dtype=torch.int64), torch.tensor([])

        pdU_indices_total = [] 
        pdU_values_total = []
        pdU_2_indices_total = []
        pdU_2_values_total = []

        index_now = 0
        batch_size = 10000
        while True:
            index_remain = index_remain_total[index_now:index_now+batch_size]
            if index_remain.shape[0] == 0:
                break

            point_pairs = self._point_pairs[:, index_remain]
            num_p = index_remain.shape[0]

            # if index_remain.shape[0] > 0:
            #     print('  Contact pairs: ', index_remain.shape[0])

            # Filter all variables
            MM = MM0[:, :, index_remain]
            E1 = E10[:, index_remain]
            E2 = E20[:, index_remain]
            T = T0[:, :, index_remain]
            L = L0[:, :, index_remain]
            dy = dy0[:, :, index_remain]
            dn = dn0[:, :, index_remain]
            f = f0[:, :, index_remain]
            g = g0[:, :, index_remain]
            h = h0[:, :, index_remain]
            weight = weight0[:, :, index_remain]

            # Calculate derivatives for both surfaces
            # Surface 1 derivatives
            n1dN1 = torch.einsum('ij, ge->geij', torch.eye(3), 1/nnorm1) + \
                torch.einsum('gei, gej, ge->geij', n1, n1, -1/nnorm1)
            n1dN1_2 = torch.einsum('ij, gek, ge->geijk', torch.eye(3), n1, -1/nnorm1**2) + \
                torch.einsum('geik, gej, ge->geijk', n1dN1, n1, -1/nnorm1) + \
                torch.einsum('gei, gejk, ge->geijk', n1, n1dN1, -1/nnorm1) + \
                torch.einsum('gei, gej, gek, ge->geijk', n1, n1, n1, 1/nnorm1**2)
            
            y1dUe = self.surface_element1.shape_function_gaussian[0]
            
            epsilon = torch.zeros([3, 3, 3])
            epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
            epsilon[1, 0, 2] = epsilon[2, 1, 0] = epsilon[0, 2, 1] = -1

            N1dUe = torch.einsum('ijl, geja->geial', epsilon, 
                                torch.einsum('gei, ga->geia', NR1[:, :, 0], 
                                            self.surface_element1.shape_function_gaussian[1][:, 1]) - 
                                torch.einsum('gei, ga->geia', NR1[:, :, 1], 
                                            self.surface_element1.shape_function_gaussian[1][:, 0]))
            
            N1dUe_2 = torch.einsum('ipl, gab->gialbp', epsilon, 
                                torch.einsum('gb,ga->gab', self.surface_element1.shape_function_gaussian[1][:, 0], self.surface_element1.shape_function_gaussian[1][:, 1])-
                                torch.einsum('gb,ga->gab', self.surface_element1.shape_function_gaussian[1][:, 1], self.surface_element1.shape_function_gaussian[1][:, 0]))

            n1dUe = torch.einsum('geij, geial->gejal', n1dN1, N1dUe)
            n1dUe_2 = torch.einsum('geijk, geial, gekbp->gejalbp', n1dN1_2, N1dUe, N1dUe) + \
                    torch.einsum('geij, gialbp->gejalbp', n1dN1, N1dUe_2)

            e1dUe = torch.zeros([num_g1, num_e1, 2, 3, num_n1, 3])
            e1dUe[:, :, 1] = n1dUe
            e1dUe[:, :, 0, 0, :, 0] = y1dUe[:, None, :]
            e1dUe[:, :, 0, 1, :, 1] = y1dUe[:, None, :]
            e1dUe[:, :, 0, 2, :, 2] = y1dUe[:, None, :]

            e1dUe_2 = torch.zeros([num_g1, num_e1, 2, 3, num_n1, 3, num_n1, 3])
            e1dUe_2[:, :, 1] = n1dUe_2

            # Surface 2 derivatives
            n2dN2 = torch.einsum('ij, ge->geij', torch.eye(3), 1/nnorm2) + \
                torch.einsum('gei, gej, ge->geij', n2, n2, -1/nnorm2)
            n2dN2_2 = torch.einsum('ij, gek, ge->geijk', torch.eye(3), n2, -1/nnorm2**2) + \
                torch.einsum('geik, gej, ge->geijk', n2dN2, n2, -1/nnorm2) + \
                torch.einsum('gei, gejk, ge->geijk', n2, n2dN2, -1/nnorm2) + \
                torch.einsum('gei, gej, gek, ge->geijk', n2, n2, n2, 1/nnorm2**2)
            
            y2dUe = self.surface_element2.shape_function_gaussian[0]

            N2dUe = torch.einsum('ijl, geja->geial', epsilon, 
                                torch.einsum('gei, ga->geia', NR2[:, :, 0], 
                                            self.surface_element2.shape_function_gaussian[1][:, 1]) - 
                                torch.einsum('gei, ga->geia', NR2[:, :, 1], 
                                            self.surface_element2.shape_function_gaussian[1][:, 0]))
            N2dUe_2 = torch.einsum('ipl, gab->gialbp', epsilon, 
                                torch.einsum('gb,ga->gab', self.surface_element2.shape_function_gaussian[1][:, 0], self.surface_element2.shape_function_gaussian[1][:, 1])-
                                torch.einsum('gb,ga->gab', self.surface_element2.shape_function_gaussian[1][:, 1], self.surface_element2.shape_function_gaussian[1][:, 0]))


            n2dUe = torch.einsum('geij, geial->gejal', n2dN2, N2dUe)
            n2dUe_2 = torch.einsum('geijk, geial, gekbp->gejalbp', n2dN2_2, N2dUe, N2dUe) + \
                    torch.einsum('geij, gialbp->gejalbp', n2dN2, N2dUe_2)

            e2dUe = torch.zeros([num_g2, num_e2, 2, 3, num_n2, 3])
            e2dUe[:, :, 1] = n2dUe
            e2dUe[:, :, 0, 0, :, 0] = y2dUe[:, None, :]
            e2dUe[:, :, 0, 1, :, 1] = y2dUe[:, None, :]
            e2dUe[:, :, 0, 2, :, 2] = y2dUe[:, None, :]

            e2dUe_2 = torch.zeros([num_g2, num_e2, 2, 3, num_n2, 3, num_n2, 3])
            e2dUe_2[:, :, 1] = n2dUe_2

            # Calculate penalty derivatives (similar to self-contact but for two surfaces)
            # g = torch.exp(D * self.penalty_factor_f) * self.penalty_distance_f
            gdD = (self.penalty_factor_f) * g
            gdD_2 = (self.penalty_factor_f**2) * g

            gdE = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3])
            gdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', gdD / 2, dn)
            gdE[:, :, :, 1, 0, :] = -gdE[:, :, :, 0, 0, :]
            gdE[:, :, :, 0, 1, :] = torch.einsum('gGp, gGpi->gGpi', gdD / 2, dy)
            gdE[:, :, :, 1, 1, :] = -gdE[:, :, :, 0, 1, :]

            gdE_2 = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3, 2, 2, 3])
            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dn, dn)
            gdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
            gdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dn, dy) + \
                    torch.einsum('gGp, ij->gGpij', gdD / 2, torch.eye(3))
            gdE_2[:, :, :, 0, 0, :, 0, 1, :] = tmp
            gdE_2[:, :, :, 0, 0, :, 1, 1, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 0, 1, :] = -tmp
            gdE_2[:, :, :, 1, 0, :, 1, 1, :] = tmp

            tmp = tmp.permute([0, 1, 2, 4, 3])
            gdE_2[:, :, :, 0, 1, :, 0, 0, :] = tmp
            gdE_2[:, :, :, 0, 1, :, 1, 0, :] = -tmp
            gdE_2[:, :, :, 1, 1, :, 0, 0, :] = -tmp
            gdE_2[:, :, :, 1, 1, :, 1, 0, :] = tmp

            temp = torch.einsum('gGp, gGpi, gGpj->gGpij', gdD_2 / 4, dy, dy)
            gdE_2[:, :, :, 0, 1, :, 0, 1, :] = temp
            gdE_2[:, :, :, 1, 1, :, 0, 1, :] = -temp
            gdE_2[:, :, :, 0, 1, :, 1, 1, :] = -temp
            gdE_2[:, :, :, 1, 1, :, 1, 1, :] = temp

            fdM = -30*MM**2*(MM-1)**2 / (self.penalty_start_g-self.penalty_end_g)
            fdM_2 = 60*MM*(MM-1)*(2*MM-1) / (self.penalty_start_g-self.penalty_end_g)**2
            fdM[MM>=1] = 0 
            fdM[MM<=0] = 0
            fdM_2[MM>=1] = 0 
            fdM_2[MM<=0] = 0
            # M = (E[:, None, :, 0, 1, :] * E[None, :, :, 1, 1, :]).sum(dim=-1)

            fdE = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3])
            fdE[:, :, :, 0, 1, :] = torch.einsum('gGp, Gpi->gGpi', fdM, E2[:, :, 1, :])
            fdE[:, :, :, 1, 1, :] = torch.einsum('gGp, gpi->gGpi', fdM, E1[:, :, 1, :])

            fdE_2 = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3, 2, 2, 3])
            fdE_2[:, :, :, 0, 1, :, 0, 1, :] = torch.einsum('gGp, Gpi, Gpj->gGpij', fdM_2, E2[:, :, 1, :], E2[:, :, 1, :])
            fdE_2[:, :, :, 0, 1, :, 1, 1, :] = torch.einsum('gGp, ij->gGpij', fdM, torch.eye(3)) + \
                                                torch.einsum('gGp, Gpi, gpj->gGpij', fdM_2, E2[:, :, 1, :], E1[:, :, 1, :])
            fdE_2[:, :, :, 1, 1, :, 1, 1, :] = torch.einsum('gGp, gpi, gpj->gGpij', fdM_2, E1[:, :, 1, :], E1[:, :, 1, :])
            fdE_2[:, :, :, 1, 1, :, 0, 1, :] = torch.einsum('gGp, ij->gGpij', fdM, torch.eye(3)) + \
            torch.einsum('gGp, gpi, Gpj->gGpij', fdM_2, E1[:, :, 1, :], E2[:, :, 1, :])

            hdE = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3])

            # L = dy.norm(dim=-1)
            # T = (self._penalty_distance - L) / (0.5 * self._penalty_distance)
            # T = T.clamp(0, 1)
            # h = T**3 * (6*T**2 - 15*T + 10)
            Lddy = torch.einsum('gGpi, gGp->gGpi', dy, 1/L)
            Lddy_2 = torch.einsum('ij, gGp->gGpij', torch.eye(3), 1/L) + torch.einsum('gGpi, gGpj, gGp->gGpij', dy, Lddy, -1/L**2)
            hdL = -30*T**2*(T-1)**2 / (self.penalty_ratio_h * self.penalty_threshold_h)
            hdL_2 = 60*T*(T-1)*(2*T-1) / (self.penalty_ratio_h * self.penalty_threshold_h)**2
            hdL[T>=1] = 0
            hdL[T<=0] = 0
            hdL_2[T>=1] = 0
            hdL_2[T<=0] = 0
            hdE[:, :, :, 0, 0, :] = torch.einsum('gGp, gGpi->gGpi', hdL, Lddy)
            hdE[:, :, :, 1, 0, :] = -hdE[:, :, :, 0, 0, :]

            hdE_2 = torch.zeros([num_g1, num_g2, num_p, 2, 2, 3, 2, 2, 3])
            tmp = torch.einsum('gGp, gGpi, gGpj->gGpij', hdL_2, Lddy, Lddy) + \
                    torch.einsum('gGp, gGpij->gGpij', hdL, Lddy_2)
            hdE_2[:, :, :, 0, 0, :, 0, 0, :] = tmp
            hdE_2[:, :, :, 0, 0, :, 1, 0, :] = -tmp
            hdE_2[:, :, :, 1, 0, :, 0, 0, :] = -tmp
            hdE_2[:, :, :, 1, 0, :, 1, 0, :] = tmp

            pdE = torch.einsum('gGpmxi, gGp, gGp->gGpmxi', fdE, g, h) + \
                torch.einsum('gGp, gGpmxi, gGp->gGpmxi', f, gdE, h) + \
                torch.einsum('gGp, gGp, gGpmxi->gGpmxi', f, g, hdE)
            
            pdE = pdE * weight[:, :, :, None, None, None]

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
            
            pdE_2 = pdE_2 * weight[:, :, :, None, None, None, None, None, None]

            # Calculate force contributions
            pdEsum0 = pdE.sum(0)
            pdEsum1 = pdE.sum(1)

            pdUe_values1 = torch.einsum('gpxi, gpxial->pal', pdEsum1[:, :, 0], e1dUe[:, point_pairs[0]])
            pdUe_values2 = torch.einsum('Gpxi, Gpxial->pal', pdEsum0[:, :, 1], e2dUe[:, point_pairs[1]])

            pdU_values = torch.cat([pdUe_values1.flatten(), pdUe_values2.flatten()], dim=0)

            tri_ind = point_pairs
            index_start1 = self._assembly.RGC_list_indexStart[instance1._RGC_index]
            index_start2 = self._assembly.RGC_list_indexStart[instance2._RGC_index]

            pdU_indices1 = self.surface_element1._elems[tri_ind[0]].to(torch.int64)
            pdU_indices1 = torch.stack([pdU_indices1*3, pdU_indices1*3+1, pdU_indices1*3+2], dim=-1) + index_start1
            
            pdU_indices2 = self.surface_element2._elems[tri_ind[1]].to(torch.int64)
            pdU_indices2 = torch.stack([pdU_indices2*3, pdU_indices2*3+1, pdU_indices2*3+2], dim=-1) + index_start2
            
            pdU_indices = torch.cat([pdU_indices1.flatten(), pdU_indices2.flatten()], dim=0).to(torch.get_default_device())

            if if_onlyforce:
                
                return pdU_indices.flatten(), -pdU_values.flatten()


            # For stiffness matrix, return simplified version (full implementation would be very complex)
            # Return empty stiffness for now
            pdUe_2_values00 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(1)[:, :, 0, :, :, 0], e1dUe[:, point_pairs[0]], e1dUe[:, point_pairs[0]]) + \
                                torch.einsum('gpxi, gpxialbL->palbL', pdEsum1[:, :, 0], e1dUe_2[:, point_pairs[0]])
            
            pdUe_2_values01 = torch.einsum('gGpxiyj, gpxial, GpyjbL->palbL', pdE_2[:, :, :, 0, :, :, 1], e1dUe[:, point_pairs[0]], e2dUe[:, point_pairs[1]])

            pdUe_2_values10 = torch.einsum('gGpxiyj, Gpxial, gpyjbL->palbL', pdE_2[:, :, :, 1, :, :, 0], e2dUe[:, point_pairs[1]], e1dUe[:, point_pairs[0]])
            
            pdUe_2_values11 = torch.einsum('gpxiyj, gpxial, gpyjbL->palbL', pdE_2.sum(0)[:, :, 1, :, :, 1], e2dUe[:, point_pairs[1]], e2dUe[:, point_pairs[1]]) + \
                                torch.einsum('gpxi, gpxialbL->palbL', pdEsum0[:, :, 1], e2dUe_2[:, point_pairs[1]])

            pdU_2_values = torch.cat([pdUe_2_values00.flatten(), pdUe_2_values01.flatten(), pdUe_2_values10.flatten().flatten(), pdUe_2_values11.flatten()], dim=0)

            # Build indices


            

            pdU_2_indices00 = torch.stack([
                self.surface_element1._elems[tri_ind[0]].reshape([num_p, num_n1, 1, 1, 1]).repeat([1, 1, 3, num_n1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n1, 1, num_n1, 3]),
                self.surface_element1._elems[tri_ind[0]].reshape([num_p, 1, 1, num_n1, 1]).repeat([1, num_n1, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n1, 3, num_n1, 1]),
            ]).reshape([4, -1])
            pdU_2_indices00 = torch.stack([pdU_2_indices00[0]*3+pdU_2_indices00[1], pdU_2_indices00[2]*3+pdU_2_indices00[3]], dim=0)
            pdU_2_indices00[0] += index_start1
            pdU_2_indices00[1] += index_start1

            pdU_2_indices01 = torch.stack([
                self.surface_element1._elems[tri_ind[0]].reshape([num_p, num_n1, 1, 1, 1]).repeat([1, 1, 3, num_n2, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n1, 1, num_n2, 3]),
                self.surface_element2._elems[tri_ind[1]].reshape([num_p, 1, 1, num_n2, 1]).repeat([1, num_n1, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n1, 3, num_n2, 1]),
            ]).reshape([4, -1])
            pdU_2_indices01 = torch.stack([pdU_2_indices01[0]*3+pdU_2_indices01[1], pdU_2_indices01[2]*3+pdU_2_indices01[3]], dim=0)
            pdU_2_indices01[0] += index_start1
            pdU_2_indices01[1] += index_start2

            pdU_2_indices10 = torch.stack([
                self.surface_element2._elems[tri_ind[1]].reshape([num_p, num_n2, 1, 1, 1]).repeat([1, 1, 3, num_n1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n2, 1, num_n1, 3]),
                self.surface_element1._elems[tri_ind[0]].reshape([num_p, 1, 1, num_n1, 1]).repeat([1, num_n2, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n2, 3, num_n1, 1]),
            ]).reshape([4, -1])
            pdU_2_indices10 = torch.stack([pdU_2_indices10[0]*3+pdU_2_indices10[1], pdU_2_indices10[2]*3+pdU_2_indices10[3]], dim=0)
            pdU_2_indices10[0] += index_start2
            pdU_2_indices10[1] += index_start1

            pdU_2_indices11 = torch.stack([
                self.surface_element2._elems[tri_ind[1]].reshape([num_p, num_n2, 1, 1, 1]).repeat([1, 1, 3, num_n2, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 3, 1, 1]).repeat([num_p, num_n2, 1, num_n2, 3]),
                self.surface_element2._elems[tri_ind[1]].reshape([num_p, 1, 1, num_n2, 1]).repeat([1, num_n2, 3, 1, 3]),
                torch.arange(3, device=tri_ind.device).reshape([1, 1, 1, 1, 3]).repeat([num_p, num_n2, 3, num_n2, 1]),
            ]).reshape([4, -1])
            pdU_2_indices11 = torch.stack([pdU_2_indices11[0]*3+pdU_2_indices11[1], pdU_2_indices11[2]*3+pdU_2_indices11[3]], dim=0)
            pdU_2_indices11[0] += index_start2
            pdU_2_indices11[1] += index_start2
            

            pdU_2_indices = torch.cat([pdU_2_indices00, pdU_2_indices01, pdU_2_indices10, pdU_2_indices11], dim=1).to(torch.get_default_device())

            index_now += batch_size

            pdU_indices_total.append(pdU_indices)
            pdU_values_total.append(pdU_values)
            pdU_2_indices_total.append(pdU_2_indices)
            pdU_2_values_total.append(pdU_2_values)

        pdU_indices = torch.cat(pdU_indices_total, dim=0)
        pdU_values = torch.cat(pdU_values_total, dim=0)
        pdU_2_indices = torch.cat(pdU_2_indices_total, dim=1)
        pdU_2_values = torch.cat(pdU_2_values_total, dim=0)

        return pdU_indices.flatten(), -pdU_values.flatten(), pdU_2_indices, -pdU_2_values

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        instance1 = self._assembly.get_instance(self.instance_name1)
        instance2 = self._assembly.get_instance(self.instance_name2)
        RGC_remain_index[instance1._RGC_index][self.surface_element1._elems.flatten().unique().cpu()] = True
        RGC_remain_index[instance2._RGC_index][self.surface_element2._elems.flatten().unique().cpu()] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseLoad.BodyForce =========#
class BodyForce(BaseLoad):

    def __init__(self, instance_name: str, element_name: str, force_density: list[float] = [0.0, 0.0, -9.81e-6]) -> None:
        """
        Initialize the body force load.
        
        Args:
            force_density (list[float]): The body force density vector [fx, fy, fz]. (unit: force per unit volume)
        """
        super().__init__()
        self._parameters = torch.tensor(force_density, dtype=torch.float64)
        
        self._element_name = element_name

        self._instance_name = instance_name

        self._pdU_indices: torch.Tensor
        self._pdU_values: torch.Tensor

        self._element: Element_3D

        self._instance_RGC_index: int

    def initialize(self, assembly):
        super().initialize(assembly)
        
        # Collect all element sets and their elements
        self._element = assembly.get_instance(self._instance_name).elems[self._element_name]

        instance = assembly.get_instance(self._instance_name)    

        self._pdU_indices = torch.stack([self._element._elems*3, self._element._elems*3+1, self._element._elems*3+2], dim=-1).to(instance.nodes.device).to(torch.int64) + instance._index_start
        self._pdU_values = torch.einsum('i, ge, gea->eai', self.force_density, self._element.gaussian_weight, self._element.shape_function_d0_gaussian).flatten()
        self._instance_RGC_index = instance._RGC_index

    @property
    def force_density(self) -> torch.Tensor:
        """
        Get the body force density vector.
        """
        return self._parameters
    
    @force_density.setter
    def force_density(self, value: list[float] | torch.Tensor) -> None:
        """
        Set the body force density vector.
        """
        if isinstance(value, list):
            self._parameters = torch.tensor(value, dtype=torch.float64)
        else:
            self._parameters = value.to(torch.float64)
    def get_stiffness(self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the body force vector. Body forces don't contribute to stiffness matrix.
        
        Args:
            RGC (list[torch.Tensor]): Current configuration.
            
        Returns:
            tuple: (F_indices, F_values, K_indices, K_values)
                - F_indices: Indices for force vector
                - F_values: Force values distributed from elements to nodes
                - K_indices: Empty tensor (body forces don't affect stiffness)
                - K_values: Empty tensor (body forces don't affect stiffness)
        """
        # Current node positions for volume calculation

        if if_onlyforce:
            return (self._pdU_indices.flatten(), self._pdU_values)

        return (self._pdU_indices.flatten(), 
                self._pdU_values, 
                torch.zeros([2, 0], dtype=torch.int64, device=self._pdU_values.device), 
                torch.zeros([0], device=self._pdU_values.device))

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Get the body force potential energy: U = -∫(f·r)dV
        
        Args:
            RGC (list[torch.Tensor]): Current configuration.
            
        Returns:
            torch.Tensor: Body force potential energy.
        """
        # Current node positions
        U = RGC[self._instance_RGC_index] + self._assembly.get_instance(self._instance_name).nodes
        elems = self._element
        displacement_gaussian = torch.zeros(elems._num_gaussian, elems._elems.shape[0], 3)
        for i in range(elems.num_nodes_per_elem):
            displacement_gaussian = displacement_gaussian + torch.einsum('ge, ei->gei', elems.shape_function_d0_gaussian[:, :, i], U[elems._elems[:, i]])

        potential_energy = torch.einsum('gei, i, ge->', displacement_gaussian, self.force_density, elems.gaussian_weight)

        return potential_energy

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Mark degrees of freedom that are affected by body forces.
        
        Args:
            RGC_remain_index (list[np.ndarray]): Current DOF activation flags.
            
        Returns:
            list[np.ndarray]: Updated DOF activation flags.
        """
        # Body forces affect nodes that belong to elements with volume
        RGC_remain_index[0][self._element._elems.flatten().cpu()] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseLoad.Spring_RP_RP =========#
class Spring_RP_RP(BaseLoad):
    """
    A nonlinear axial spring connecting two reference points (RP-RP).

    Parameters:
        rp_name1: name of first reference point
        rp_name2: name of second reference point
        k: spring stiffness
        rest_length (optional): rest length L0; defaults to initial distance between RPs
    """

    def __init__(self, rp_name1: str, rp_name2: str, k: float, rest_length: float | None = None) -> None:
        super().__init__()
        self.rp_name1 = rp_name1
        self.rp_name2 = rp_name2
        
        rl = rest_length if rest_length is not None else -1.0
        self._parameters = torch.tensor([k, rl], dtype=torch.float64)

    @property
    def k(self) -> float:
        return self._parameters[0]
    
    @k.setter
    def k(self, value: float) -> None:
        self._parameters[0] = value

    @property
    def rest_length(self) -> float:
        return self._parameters[1]
    
    @rest_length.setter
    def rest_length(self, value: float) -> None:
        self._parameters[1] = value

        # indices cache
        self._rp_index1: int | None = None
        self._rp_index2: int | None = None
        self._idx_tr1: torch.Tensor | None = None
        self._idx_tr2: torch.Tensor | None = None

    def initialize(self, assembly):
        super().initialize(assembly)
        rp1 = assembly.get_reference_point(self.rp_name1)
        rp2 = assembly.get_reference_point(self.rp_name2)
        self._rp_index1 = rp1._RGC_index
        self._rp_index2 = rp2._RGC_index

        s1 = assembly.RGC_list_indexStart[self._rp_index1]
        s2 = assembly.RGC_list_indexStart[self._rp_index2]
        self._idx_tr1 = torch.arange(s1, s1 + 3, device=assembly.device, dtype=torch.int64)
        self._idx_tr2 = torch.arange(s2, s2 + 3, device=assembly.device, dtype=torch.int64)

        # Default rest length from initial geometry if not provided
        if self.rest_length < 0:
            p1 = rp1.node.to(assembly.device).to(torch.get_default_dtype())
            p2 = rp2.node.to(assembly.device).to(torch.get_default_dtype())
            self.rest_length = torch.linalg.norm(p2 - p1).item()

    def get_stiffness(self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs):
        # Current positions of the two RPs
        rp1 = self._assembly.get_reference_point(self.rp_name1)
        rp2 = self._assembly.get_reference_point(self.rp_name2)
        x1 = rp1.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index1][:3]
        x2 = rp2.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index2][:3]

        d = x2 - x1
        l = torch.linalg.norm(d)
        eps = 1e-16
        if l.item() < eps:
            # No defined direction; no force or stiffness
            F_indices = torch.cat([self._idx_tr1, self._idx_tr2], dim=0)
            F_values = torch.zeros(6, dtype=x1.dtype, device=x1.device)
            if if_onlyforce:
                return F_indices, F_values
            K_indices = torch.zeros((2, 0), dtype=torch.int64, device=x1.device)
            K_values = torch.zeros(0, dtype=x1.dtype, device=x1.device)
            return F_indices, F_values, K_indices, K_values

        n = d / l
        f = self.k * (l - self.rest_length) * n
        f1 = f
        f2 = -f

        F_indices = torch.cat([self._idx_tr1, self._idx_tr2], dim=0)
        F_values = torch.cat([f1, f2], dim=0)

        if if_onlyforce:
            return F_indices, F_values

        # Tangent blocks (3x3)
        K_block = _spring_tangent_block(d, self.k, self.rest_length)
        # Build COO indices for 6x6 blocks
        rows11 = self._idx_tr1.repeat_interleave(3)
        cols11 = self._idx_tr1.repeat(3)
        rows12 = self._idx_tr1.repeat_interleave(3)
        cols12 = self._idx_tr2.repeat(3)
        rows21 = self._idx_tr2.repeat_interleave(3)
        cols21 = self._idx_tr1.repeat(3)
        rows22 = self._idx_tr2.repeat_interleave(3)
        cols22 = self._idx_tr2.repeat(3)

        K_indices = torch.stack([
            torch.cat([rows11, rows12, rows21, rows22], dim=0),
            torch.cat([cols11, cols12, cols21, cols22], dim=0)
        ], dim=0)

        K_values = torch.cat([
            (-K_block).reshape(-1),  # K11
            (K_block).reshape(-1),   # K12
            (K_block).reshape(-1),   # K21
            (-K_block).reshape(-1)   # K22
        ], dim=0)

        return F_indices, F_values, K_indices, K_values

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        rp1 = self._assembly.get_reference_point(self.rp_name1)
        rp2 = self._assembly.get_reference_point(self.rp_name2)
        x1 = rp1.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index1][:3]
        x2 = rp2.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index2][:3]
        l = torch.linalg.norm(x2 - x1)
        # Negative sign so Assembly._total_Potential_Energy adds spring energy (internal-like)
        return -0.5 * self.k * (l - self.rest_length) ** 2

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        RGC_remain_index[self._rp_index1][:3] = True
        RGC_remain_index[self._rp_index2][:3] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseLoad.Spring_RP_Point =========#
class Spring_RP_Point(BaseLoad):
    """
    A nonlinear axial spring connecting one reference point to a fixed point in space (RP-Point).

    Parameters:
        rp_name: name of the reference point
        point: [x, y, z] fixed spatial point
        k: spring stiffness
        rest_length (optional): rest length L0; defaults to initial distance between RP and point
    """

    def __init__(self, rp_name: str, point: list[float], k: float, rest_length: float = None) -> None:
        
        super().__init__()
        self.rp_name = rp_name

        self.point = torch.tensor(point)
        self.k = torch.tensor(k)
        self.rest_length = None if rest_length is None else torch.tensor(rest_length)

        self._rp_index: int | None = None
        self._idx_tr: torch.Tensor | None = None

    def initialize(self, assembly):
        super().initialize(assembly)
        rp = assembly.get_reference_point(self.rp_name)
        self._rp_index = rp._RGC_index
        s = assembly.RGC_list_indexStart[self._rp_index]
        self._idx_tr = torch.arange(s, s + 3, device=assembly.device, dtype=torch.int64)
        # Materialize _P on the correct device/dtype via property setter
        self.point = torch.tensor(self.point)

        # Default rest length from initial geometry if not provided
        if self.rest_length is None:
            x0 = rp.node.to(assembly.device).to(torch.get_default_dtype())
            self.rest_length = torch.linalg.norm(self.point - x0)

    def get_stiffness(self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs):

        if type(self.point) != torch.Tensor:
            self.point = torch.tensor(self.point)
        

        rp = self._assembly.get_reference_point(self.rp_name)
        x = rp.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index][:3]
        d = self.point - x
        l = torch.linalg.norm(d)

        eps = 1e-16
        if l.item() < eps:
            F_indices = self._idx_tr
            F_values = torch.zeros(3, dtype=x.dtype, device=x.device)
            if if_onlyforce:
                return F_indices, F_values
            K_indices = torch.zeros((2, 0), dtype=torch.int64, device=x.device)
            K_values = torch.zeros(0, dtype=x.dtype, device=x.device)
            return F_indices, F_values, K_indices, K_values

        n = d / l
        f = self.k * (l - self.rest_length) * n
        F_indices = self._idx_tr
        F_values = f

        if if_onlyforce:
            return F_indices, F_values

        K_block = _spring_tangent_block(d, self.k, self.rest_length)
        rows = self._idx_tr.repeat_interleave(3)
        cols = self._idx_tr.repeat(3)
        K_indices = torch.stack([rows, cols], dim=0)
        K_values = (-K_block).reshape(-1)  # derivative wrt RP coords
        return F_indices, F_values, K_indices, K_values

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        if type(self.point) != torch.Tensor:
            self.point = torch.tensor(self.point)
            
        rp = self._assembly.get_reference_point(self.rp_name)
        x = rp.node.to(self._assembly.device).to(torch.get_default_dtype()) + RGC[self._rp_index][:3]
        l = torch.linalg.norm(self.point - x)
        # Negative sign so Assembly._total_Potential_Energy adds spring energy (internal-like)
        return -0.5 * self.k * (l - self.rest_length) ** 2

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        RGC_remain_index[self._rp_index][:3] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseLoad.Penalty_DoF =========#
class Penalty_DoF(BaseLoad):
    """
    Penalize one DoF in one object's RGC segment to track a target value using quadratic penalty energy.

    Parameters:
        obj_name: object name in assembly
        s: local flattened DoF index in the selected object's RGC segment
        target: desired value for the selected DoF
        k: penalty coefficient
        obj_type: one of {'auto', 'instance', 'rp', 'load', 'constraint'}
    """

    _serialized_attributes = ['obj_name', 's', 'target', 'k', '_parameters', 'obj_type']

    def __init__(
        self,
        obj_name: str,
        s: int,
        target: float,
        k: float,
        obj_type: str = "auto",
    ) -> None:
        super().__init__()

        self.obj_name = obj_name
        self.s = int(s)
        self.obj_type = obj_type

        # [k, target]
        self._parameters = torch.tensor([k, target], dtype=torch.float64)
        self._indices_force: torch.Tensor | None = None

        # Located in initialize from object + local index s
        self._rgc_list_index: int | None = None
        self._rgc_local_flat_index: int | None = None
        self._global_s: int | None = None



    @property
    def k(self) -> torch.Tensor:
        return self._parameters[0]

    @k.setter
    def k(self, value: float) -> None:
        self._parameters[0] = value

    @property
    def target(self) -> torch.Tensor:
        return self._parameters[1]

    @target.setter
    def target(self, value: float) -> None:
        self._parameters[1] = value

    def _resolve_obj(self, assembly):
        containers = {
            "instance": assembly._instances,
            "rp": assembly._reference_points,
            "load": assembly._loads,
            "constraint": assembly._constraints,
        }

        if self.obj_type == "auto":
            found = []
            for kind, data in containers.items():
                if self.obj_name in data:
                    found.append((kind, data[self.obj_name]))

            if len(found) == 0:
                raise ValueError(
                    f"Object '{self.obj_name}' not found in instance/rp/load/constraint."
                )
            if len(found) > 1:
                kinds = [item[0] for item in found]
                raise ValueError(
                    f"Object name '{self.obj_name}' is ambiguous in {kinds}. Please set obj_type explicitly."
                )
            return found[0][1]

        if self.obj_type not in containers:
            raise ValueError("obj_type must be one of {'auto', 'instance', 'rp', 'load', 'constraint'}")

        if self.obj_name not in containers[self.obj_type]:
            raise ValueError(f"{self.obj_type} '{self.obj_name}' not found in assembly.")

        return containers[self.obj_type][self.obj_name]

    def _locate_object_local_s(self, assembly):
        obj = self._resolve_obj(assembly)
        rgc_idx = obj._RGC_index

        if rgc_idx is None:
            raise ValueError(f"Object '{self.obj_name}' does not have a valid _RGC_index.")

        seg_size = int(np.prod(assembly._RGC_size[rgc_idx]))
        if self.s < 0 or self.s >= seg_size:
            raise ValueError(
                f"local s={self.s} out of range for object '{self.obj_name}', segment size={seg_size}"
            )

        global_s = int(assembly.RGC_list_indexStart[rgc_idx]) + self.s
        return rgc_idx, self.s, global_s

    def initialize(self, assembly):
        super().initialize(assembly)

        rgc_idx, local_flat, global_s = self._locate_object_local_s(assembly)
        self._rgc_list_index = rgc_idx
        self._rgc_local_flat_index = local_flat
        self._global_s = global_s

        self._indices_force = torch.tensor([self._global_s], dtype=torch.int64, device=assembly.device)

    def _get_params_like(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k.to(device=ref.device, dtype=ref.dtype)
        target = self.target.to(device=ref.device, dtype=ref.dtype)
        return k, target

    def _get_s_now(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return RGC[self._rgc_list_index].reshape(-1)[self._rgc_local_flat_index]

    def get_stiffness(
        self,
        RGC: list[torch.Tensor],
        if_onlyforce: bool = False,
        *args,
        **kwargs,
    ):
        s_now = self._get_s_now(RGC)
        k, target = self._get_params_like(s_now)

        # f is defined like spring loads so Assembly residual uses -f.
        f = k * (target - s_now)
        F_indices = self._indices_force
        F_values = f.reshape(1)

        if if_onlyforce:
            return F_indices, F_values

        # Tangent of f wrt selected DoF: df/ds = -k
        K_indices = torch.stack([F_indices, F_indices], dim=0)
        K_values = (-k).reshape(1)
        return F_indices, F_values, K_indices, K_values

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        s_now = self._get_s_now(RGC)
        k, target = self._get_params_like(s_now)
        # Negative sign so Assembly._total_Potential_Energy adds penalty energy.
        return -0.5 * k * (s_now - target) ** 2

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        arr = RGC_remain_index[self._rgc_list_index].reshape(-1)
        arr[self._rgc_local_flat_index] = True
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseConstraint =========#
class BaseConstraint(BaseObj):
    """
    Constraints base class
    """

    def __init__(self) -> None:
        """
        Initialize the Constraints_Base class.
        """
        super().__init__()

    def initialize(self, assembly):
        super().initialize(assembly)


    def modify_R_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
                   K_indices: torch.Tensor = None, K_values: torch.Tensor = None, if_onlyforce: bool = False, *args, **kwargs):

        R = torch.sparse_coo_tensor(indices=[[]],
                                    values=[],
                                    size=[self._assembly.RGC_list_indexStart[-1]])
        if if_onlyforce:
            return R
        return R, torch.zeros([2, 0], dtype=torch.int64), torch.zeros([0])


    def modify_mass_matrix(self, mass_indices: torch.Tensor, mass_values: torch.Tensor, RGC: list[torch.Tensor]):
        return torch.zeros([2, 0], dtype=torch.int64), torch.zeros([0])


#========= Source code for Serializable.BaseObj.BaseConstraint.Couple =========#
class Couple(BaseConstraint):

    _serialized_attributes = ['instance_name', 'set_nodes_name', 'rp_name']

    def __init__(self, instance_name: str, set_nodes_name: str, rp_name: str) -> None:
        super().__init__()
        self.instance_name = instance_name
        self.set_nodes_name = set_nodes_name
        self.rp_name = rp_name

        
        self._ref_location: torch.Tensor

        self._couple_index: int
        self._rp_index: int
        self._instance_RGC_index: int
        self._indexNodes: np.ndarray

    def initialize(self, assembly):
        super().initialize(assembly)
        self._indexNodes = np.sort(self._assembly.get_instance(self.instance_name).set_nodes[self.set_nodes_name])
        self._rp_index = self._assembly.get_reference_point(self.rp_name)._RGC_index
        self._couple_index = self._assembly.get_instance(self.instance_name)._RGC_index

        instance = self._assembly.get_instance(self.instance_name)
        self._instance_RGC_index = instance._RGC_index

        index_global = instance.nodes[self._indexNodes]
        self._ref_location = index_global - self._assembly.get_reference_point(self.rp_name).node

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply the couple constraint to the displacement vector
        """
        RGC[self._couple_index][self._indexNodes] = RGC[self._rp_index][:3] + self._rotation3d(
            RGC[self._rp_index][3:], self._ref_location) - self._ref_location

        return RGC
    
    def modify_mass_matrix(self, mass_indices, mass_values, RGC: list[torch.Tensor]):
        v = RGC[self._rp_index][:3]
        z = RGC[self._rp_index][3:]

        theta = z.norm() + 1e-20
        w = (z / theta)

        epsilon = torch.zeros([3, 3, 3])
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[1, 0, 2] = epsilon[2, 1, 0] = -1

        y = v - self._ref_location + \
            self._ref_location * torch.cos(theta) + \
            torch.einsum('ijk, j, pk->pi', epsilon, w, self._ref_location) * torch.sin(theta) + \
            torch.einsum('i,j,pj->pi', w, w, self._ref_location) * (1 - torch.cos(theta))

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self._couple_index][self._indexNodes] = False
        RGC_remain_index[self._rp_index][:] = True
        return RGC_remain_index

    def modify_R_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
                   K_indices: torch.Tensor = None, K_values: torch.Tensor = None, if_onlyforce: bool = False, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify the R and K

        Args:
            indexStart (list[int]): The starting indices for each node.
            U (list[torch.Tensor]): The displacement vector for each node.
            R (torch.Tensor): The global force vector.
            K (torch.Tensor): The global stiffness matrix.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The modified R and K tensors.
        """

        if not if_onlyforce and (K_indices is None or K_values is None):
            raise ValueError("K_indices and K_values must be provided when if_onlyforce is False")

        R_now = R0[self._assembly.RGC_list_indexStart[self._instance_RGC_index]:self._assembly.RGC_list_indexStart[self._instance_RGC_index+1]].view(-1, 3)
        Ydot, Ydot2 = self._calculate_Ydotz(RGC)

        # R
        # region
        Rrest = R_now[self._indexNodes]

        Edotv = Rrest.sum(dim=0)
        Edotz = torch.einsum('bj,bjp->p', Rrest, Ydot)

        R = torch.zeros(self._assembly.RGC_list_indexStart[-1])
        start_idx = self._assembly.RGC_list_indexStart[self._rp_index]
        R[start_idx:start_idx+3] += Edotv
        R[start_idx+3:start_idx+6] += Edotz
        # endregion

        if if_onlyforce:
            return R

        # K
        # region
        ## first, get the K of the rest part in index1

        # initial select the instance indices

        indice_max = self._assembly.RGC_list_indexStart[-1]
        indice_start = self._assembly.RGC_list_indexStart[self._couple_index]

        index = torch.where(
            torch.isin(((K_indices[1] - indice_start) // 3),
                    torch.tensor(self._indexNodes.tolist())))

        sort_index = torch.argsort((K_indices[1][index] - indice_start) // 3)

        index = index[0][sort_index]
        indice1 = K_indices[0][index]
        indice30 = K_indices[1][index] // 3
        indice3 = torch.unique_consecutive(indice30, return_inverse=True)[1]
        indice4 = K_indices[1][index] % 3

        Rdotv_indices = torch.stack([indice1, indice4], dim=0)
        Rdotv_indices_flatten = Rdotv_indices[0] * 3 + Rdotv_indices[1]
        Rdotv_values = K_values[index]
        Rdotv = torch.zeros([indice_max * 3]).scatter_add_(
            0, Rdotv_indices_flatten,
            Rdotv_values).reshape(indice_max, 3)

        Rdotz_indices = torch.stack([
            indice1.reshape(-1, 1).repeat(1, 3),
            torch.tensor([0, 1, 2]).reshape([1, 3]).repeat(
                indice1.shape[0], 1)
        ],
                                    dim=0).reshape([2, -1])
        Rdotz_indices_flatten = Rdotz_indices[0] * 3 + Rdotz_indices[1]
        Rdotz_values = (K_values[index].unsqueeze(-1) *
                        Ydot.view(-1, 3)[indice4 + indice3 * 3]).flatten()
        Rdotz = torch.zeros([indice_max * 3]).scatter_add_(
            0, Rdotz_indices_flatten,
            Rdotz_values).reshape(indice_max, 3)
        
        index_remain_dim0 = np.vstack([indice_start + self._indexNodes * 3,
                                    indice_start + self._indexNodes * 3 + 1,
                                    indice_start + self._indexNodes * 3 + 2]).T

        Edotvv = Rdotv[index_remain_dim0].sum(dim=0)
        Edotzv = torch.einsum('biq,bip->pq', Rdotv[index_remain_dim0], Ydot)

        Edotzz = torch.einsum('biq,bip->pq', Rdotz[index_remain_dim0],
                            Ydot) + torch.einsum('ai,aipq->pq', Rrest, Ydot2)
        # combine the indices and values
        indices = []
        values = []

        ## for Rv
        indice_Rv = Rdotv_indices
        index1 = indice_Rv[0]
        index2 = self._assembly.RGC_list_indexStart[self._rp_index] + indice_Rv[1]
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(Rdotv_values)
        indices.append(torch.stack([index2, index1], dim=0))
        values.append(Rdotv_values)
        ## for Rz
        indice_Rz = Rdotz_indices
        index1 = indice_Rz[0]
        index2 = self._assembly.RGC_list_indexStart[self._rp_index] + indice_Rz[1] + 3
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(Rdotz_values)
        indices.append(torch.stack([index2, index1], dim=0))
        values.append(Rdotz_values)
        ## for Edot2
        mat66 = torch.zeros([6, 6])
        mat66[:3, :3] = Edotvv
        mat66[3:, 3:] = Edotzz
        mat66[3:, :3] = Edotzv
        mat66[:3, 3:] = Edotzv.transpose(0, 1)

        indice_Edot2 = [
            torch.tensor([0, 1, 2, 3, 4, 5]).reshape(-1,
                                                    1).repeat(1, 6).flatten(),
            torch.tensor([0, 1, 2, 3, 4,
                        5]).reshape(1, -1).repeat(6, 1).flatten()
        ]
        index1 = indice_Edot2[0] + self._assembly.RGC_list_indexStart[self._rp_index]
        index2 = indice_Edot2[1] + self._assembly.RGC_list_indexStart[self._rp_index]
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(mat66.flatten())

        # combine the indices and values
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values, dim=0)
        #endregion

        return R, indices, values

    def _calculate_Ydotz(self, RGC: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        v = RGC[self._rp_index][:3]
        z = RGC[self._rp_index][3:]
        theta = z.norm() + 1e-20
        w = (z / theta)

        epsilon_indices = [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1],
                        [2, 1, 2, 0, 1, 0]]
        epsilon_values = [1, -1, -1, 1, 1, -1]


        # basic derivatives
        y = self._ref_location
        Y = v + self._rotation3d(z, y) - y

        der_theta = -y * torch.sin(theta) + w.view(
            1, 3) * (w.view(1, 3) * y).sum(dim=1).reshape(-1, 1) * torch.sin(
                theta) + torch.cross(w.view(1, 3), y, dim=1) * torch.cos(theta)

        der_theta2 = -y * torch.cos(theta) + w.view(
            1, 3) * (w.view(1, 3) * y).sum(dim=1).reshape(
                -1, 1) * (torch.cos(theta)) - torch.cross(
                    w.view(1, 3), y, dim=1) * torch.sin(theta)

        der_w = (torch.einsum('al,i->ail', y, w)) * (1 - torch.cos(theta))
        temp = (1 - torch.cos(theta)) * (w.view(1, 3) * y).sum(dim=1).flatten()
        for i in range(3):
            der_w[:, i, i] += temp
        for i in range(6):
            der_w[:, epsilon_indices[0][i],
                epsilon_indices[2][i]] -= epsilon_values[i] * torch.sin(
                    theta) * y[:, epsilon_indices[1][i]]

        der_w2 = torch.zeros([y.shape[0], 3, 3, 3])
        temp = (1 - torch.cos(theta)) * y
        for i in range(3):
            der_w2[:, i, i, :] += temp
            der_w2[:, i, :, i] += temp

        der_w_theta = (torch.einsum('al,i->ail', y, w)) * torch.sin(theta)
        temp = torch.sin(theta) * (w.view(1, 3) * y).sum(dim=1).flatten()
        for i in range(3):
            der_w_theta[:, i, i] += temp
        for i in range(6):
            der_w_theta[:, epsilon_indices[0][i],
                        epsilon_indices[2][i]] -= epsilon_values[
                            i] * torch.cos(theta) * y[:, epsilon_indices[1][i]]

        wdot = -torch.einsum('i,p->ip', z, z) / theta**3 + torch.eye(3) / theta
        thetadot = w
        wdot2 = 3 * torch.einsum('i,p,q->ipq', z, z, z) / theta**5
        temp = z / theta**3
        for i in range(3):
            wdot2[i, i, :] -= temp
            wdot2[i, :, i] -= temp
            wdot2[:, i, i] -= temp
        thetadot2 = wdot

        Ydot = torch.einsum('bjl,lp->bjp', der_w, wdot) + torch.einsum(
            'bj,p->bjp', der_theta, thetadot)

        Ydot2 = (
            torch.einsum('ai,pq->aipq', der_theta, thetadot2) +
            torch.einsum('ai, p, q->aipq', der_theta2, thetadot, thetadot) +
            torch.einsum('ail,lq,p->aipq', der_w_theta, wdot, thetadot))

        Ydot2 += (torch.einsum('ailm,lp,mq->aipq', der_w2, wdot, wdot) +
                torch.einsum('ail,lp,q->aipq', der_w_theta, wdot, thetadot) +
                torch.einsum('ail,lpq->aipq', der_w, wdot2))
        
        return Ydot, Ydot2

    def _rotation3d(self, rotation_vector: torch.Tensor,
                    vector0: torch.Tensor):
        """
        Rotate a 3D vector by a rotation vector
        :param rotation_vector: rotation vector (3,)
        :param vector0: 3D vector (n, 3)
        :return: 3D vector (n, 3)
        """
        vector0 = vector0.view(-1, 3)
        theta = torch.norm(rotation_vector) + 1e-20
        if theta == 0:
            return vector0
        else:
            rotation_vector = rotation_vector / theta
            rotation_vector = rotation_vector.view(1, 3)
            vector1 = vector0 * torch.cos(theta) + torch.cross(
                rotation_vector, vector0, dim=1) * torch.sin(
                    theta) + rotation_vector * (rotation_vector * vector0).sum(
                        dim=1).unsqueeze(-1) * (1 - torch.cos(theta))
        return vector1


#========= Source code for Serializable.BaseObj.BaseBoundary =========#
class BaseBoundary(BaseObj):
	"""
	Boundary base class. Provides the same surface API as constraints where relevant,
	but typically only modifies RGC and required DoFs (Dirichlet conditions).
	"""

	def __init__(self) -> None:
		super().__init__()

	def initialize(self, assembly):
		super().initialize(assembly)

	def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
		"""
		Modify the RGC_remain_index to deactivate constrained DoFs.
		"""
		return RGC_remain_index

	def modify_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
		"""
		Apply boundary conditions directly to the RGC values (Dirichlet), if needed.
		"""
		return RGC

	# For compatibility with Assembly constraint hooks, provide no-op stubs
	def modify_R_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
				   K_indices: torch.Tensor = None, K_values: torch.Tensor = None,
				   if_onlyforce: bool = False, *args, **kwargs):
		if if_onlyforce:
			return torch.zeros(self._assembly.RGC_list_indexStart[-1])
		return (torch.zeros(self._assembly.RGC_list_indexStart[-1]),
				torch.zeros([2, 0], dtype=torch.int64),
				torch.zeros([0]))


#========= Source code for Serializable.BaseObj.BaseBoundary.Boundary_Condition =========#
class Boundary_Condition(BaseBoundary):
    """
    Boundary condition (Dirichlet) for instances: fix selected DoFs on a set of nodes.
    """

    def __init__(self,
                 instance_name: str,
                 set_nodes_name: str,
                 indexDoF: list[int] = [0, 1, 2],
                 ) -> None:
        super().__init__()
        self.set_nodes_name = set_nodes_name
        self.instance_name = instance_name
        self.indexDoF = indexDoF
        self._constraint_index: int
        self.index_nodes: np.ndarray

    def initialize(self, assembly):
        super().initialize(assembly)
        self.index_nodes = self._assembly.get_instance(self.instance_name).set_nodes[self.set_nodes_name]
        self._constraint_index = self._assembly.get_instance(self.instance_name)._RGC_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Apply the boundary condition to the displacement vector
        """
        for i in self.indexDoF:
            RGC[self._constraint_index][self.index_nodes, i] = 0.0
        return RGC

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index by deactivating constrained DoFs
        """
        for i in self.indexDoF:
            RGC_remain_index[self._constraint_index][self.index_nodes, i] = False
        return RGC_remain_index


#========= Source code for Serializable.BaseObj.BaseBoundary.Boundary_Condition_RP =========#
class Boundary_Condition_RP(BaseBoundary):
    """
    Boundary condition for reference points (6 DoFs: UX, UY, UZ, RX, RY, RZ).
    """

    def __init__(self,
                 rp_name: str,
                 indexDoF: list[int] = [0, 1, 2, 3, 4, 5],
                 ) -> None:
        super().__init__()
        self.rp_name = rp_name
        self.indexDoF = indexDoF
        self._constraint_index: int

    def initialize(self, assembly):
        super().initialize(assembly)
        self._constraint_index = self._assembly.get_reference_point(self.rp_name)._RGC_index

    def modify_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        for i in self.indexDoF:
            RGC[self._constraint_index][i] = 0.0
        return RGC

    def set_required_DoFs(self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        for i in self.indexDoF:
            RGC_remain_index[self._constraint_index][i] = False
        return RGC_remain_index


#========= Source code for Serializable.Assembly =========#
class Assembly(Serializable):

    _serialized_attributes: list[str] = ['_parts', '_instances', '_surfaces', '_reference_points', '_loads', '_constraints', '_boundarys']

    def __init__(self):
        
        self.device = torch.zeros(1).device
        """The default device where the tensors are stored (CPU or GPU)."""

        self._parts: dict[str, Part] = {}
        """Dictionary to store parts with part names as keys and Part objects as values."""

        self._instances: dict[str, Instance] = {}
        """Dictionary to store instances with instance names as keys and Instance objects as values."""

        self._surfaces: dict[(str, str), _Surfaces] = {}
        """Dictionary to store surface sets with keys as (instance_name, set_name) and values as Surface objects."""
        
        self._reference_points: dict[str, ReferencePoint] = {}
        """Dictionary to store reference points with reference point names as keys and ReferencePoint objects as values."""
        self._loads: dict[str, loads.BaseLoad] = {}
        """Dictionary to store loads with load names as keys and Load objects as values."""
        self._constraints: dict[str, constraints.BaseConstraint] = {}
        """Dictionary to store constraints with constraint names as keys and Constraint objects as values."""
        self._boundarys: dict[str, boundarys.BaseBoundary] = {}
        """Dictionary to store boundary conditions with names as keys and Boundary objects as values."""


        self.RGC: list[torch.Tensor]
        """
        record the redundant generalized coordinates
        """

        self._RGC_size: list[tuple[int]]
        """Record the size of each RGC component
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
        self.RGC_list_indexStart: list[int] = []
        """Record the start index of the RGC\n
        """

        self.mass_matrix_indices: torch.Tensor
        """The indices of the mass matrix"""

        self.mass_matrix_values: torch.Tensor
        """The values of the mass matrix"""

    # region visualization
    def get_meshes(self, GC: torch.Tensor = None) -> dict[str, pv.PolyData]:
        """
        Get the meshes of all instances in the assembly.

        Args:
            GC (torch.Tensor, optional): The generalized coordinates to use for visualization. Defaults to None.

        Returns:
            dict[str, pv.PolyData]: A dictionary with instance names as keys and their corresponding meshes as values.
        """
        meshes = {}
        for ins_name, ins in self._instances.items():
            if GC is not None:
                RGC = self._GC2RGC(GC)
            else:
                RGC = None
            mesh = ins.get_mesh(RGC=RGC)
            meshes[ins_name] = mesh
        return meshes

    def show_ins(self, ins_name: str, GC: torch.Tensor = None, surf_name: str = None):
        """
        Visualize the specified instance.

        Args:
            ins_name (str): The name of the instance to visualize.
            GC (torch.Tensor, optional): The generalized coordinates to use for visualization. Defaults to None.
            surf_name (str, optional): The name of the surface to visualize. Defaults to None.
        """
        if GC is not None:
            RGC = self._GC2RGC(GC)
        else:
            RGC = None
        mesh = self._instances[ins_name].get_mesh(RGC=RGC, surf_name=surf_name)
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, opacity=1.0, label=ins_name)
        plotter.add_legend()
        plotter.show()

    def show_all(self, GC: torch.Tensor = None):
        """
        Visualize all instances in the assembly.
        
        before calling this function, make sure all instances have been assigned external_surface attribute.
        assembly.get_instance('instance_name').external_surface = 'surface_set_name'

        Args:
            GC (torch.Tensor, optional): The generalized coordinates to use for visualization. Defaults to None.
        """
        meshes = self.get_meshes(GC=GC)
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter()
        for ins_name, mesh in meshes.items():
            plotter.add_mesh(mesh, show_edges=True, opacity=1.0, label=ins_name)
        plotter.add_legend()
        plotter.show()
    # endregion

    # region Initialization

    def initialize(self, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            None
        """

        # region sort the parts, instances, loads, and constraints
        self._parts = dict(sorted(self._parts.items()))
        self._instances = dict(sorted(self._instances.items()))
        self._loads = dict(sorted(self._loads.items()))
        self._constraints = dict(sorted(self._constraints.items()))
        self._reference_points = dict(sorted(self._reference_points.items()))
        self._boundarys = dict(sorted(self._boundarys.items()))
        # endregion

        # region initialize the instance with the part
        for ins in self._instances.values():
            part_name = ins.part_name
            if part_name not in self._parts:
                raise ValueError(
                    f"Part '{part_name}' not found for instance '{ins}'.")
            ins.part = self._parts[part_name]
            ins._RGC_requirements = tuple(ins.part.nodes.shape)

        # region initialize the RGC

        # initialize the RGC (redundant generalized coordinate)
        self.RGC = []
        self.RGC_remain_index = []
        self.RGC_list_indexStart = [0]
        self._RGC_size = []

        for ins in self._instances.keys():
            RGC_index = self._allocate_RGC(
                size=self._instances[ins]._RGC_requirements)
            self._instances[ins].set_RGC_index(RGC_index)

        for rp in self._reference_points.keys():
            RGC_index = self._allocate_RGC(
                size=self._reference_points[rp]._RGC_requirements)
            self._reference_points[rp].set_RGC_index(RGC_index)
            self.RGC[RGC_index][-1] = 1e-5

        for f in self._loads.keys():
            RGC_index = self._allocate_RGC(
                size=self._loads[f]._RGC_requirements)
            self._loads[f].set_RGC_index(RGC_index)

        for c in self._constraints.keys():
            RGC_index = self._allocate_RGC(
                size=self._constraints[c]._RGC_requirements)
            self._constraints[c].set_RGC_index(RGC_index)

        for b in self._boundarys.keys():
            RGC_index = self._allocate_RGC(
                size=self._boundarys[b]._RGC_requirements)
            self._boundarys[b].set_RGC_index(RGC_index)

        # endregion

        # region initialize the elements, loads, and constraints

        # initialize the parts
        for part in self._parts.values():
            part.initialize()

        # initialize the instances
        for ins in self._instances.values():
            ins.initialize(self)

        # initialize the loads
        for l in self._loads.values():
            l.initialize(self)

        # initialize the constraints
        for c in self._constraints.values():
            c.initialize(self)

        # initialize the boundary conditions
        for b in self._boundarys.values():
            b.initialize(self)

        # endregion

        # region modify the RGC_remain_index
        for ins in self._instances.values():
            self.RGC_remain_index = ins.set_required_DoFs(self.RGC_remain_index)

        for f in self._loads.values():
            self.RGC_remain_index = f.set_required_DoFs(self.RGC_remain_index)

        for c in self._constraints.values():
            self.RGC_remain_index = c.set_required_DoFs(self.RGC_remain_index)

        # Finally, apply boundary conditions to deactivate Dirichlet DOFs
        for b in self._boundarys.values():
            self.RGC_remain_index = b.set_required_DoFs(self.RGC_remain_index)

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

    def initialize_dynamic(self):
            
        for ins in self._instances.values():
            ins.initialize_dynamic()

        for l in self._loads.values():
            l.initialize_dynamic()

        for c in self._constraints.values():
            c.initialize_dynamic()

        # assemble the redundant mass matrix
        mass_indices = []
        mass_values = []
        for ins in self._instances.values():
            indices_now, values_now = ins.get_mass_matrix()
            mass_indices.append(indices_now)
            mass_values.append(values_now)
        self.mass_matrix_indices = torch.cat(mass_indices, dim=1)
        self.mass_matrix_values = torch.cat(mass_values, dim=0)

    def reinitialize(self, RGC: list[torch.Tensor]):
        """
        Reinitializes the finite element analysis problem.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
        """
        self.RGC = RGC
        self.GC = self._RGC2GC(self.RGC)

        for ins in self._instances.values():
            ins.reinitialize(RGC)

        for l in self._loads.values():
            l.reinitialize(RGC)

        for c in self._constraints.values():
            c.reinitialize(RGC)
    # endregion

    # region Stiffness Matrix Assembly

    def assemble_force(self, RGC: list[torch.Tensor] = None, GC: torch.Tensor = None) -> torch.Tensor:
        
        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        #region evaluate the structural K and R
        t0 = time.time()
        R_values = []
        R_indices = []

        for ins in self._instances.keys():
            Ra_indice, Ra_values = self._instances[ins].structural_stiffness(
                RGC=RGC, if_onlyforce=True)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)
        t1 = time.time()

        ff = []
        for f in self._loads.values():
            Rf_indice, Rf_values = f.get_stiffness(
                RGC=RGC, if_onlyforce=True)
            R_values.append(-Rf_values)
            R_indices.append(Rf_indice)

            ff.append(torch.zeros(self.RGC_list_indexStart[-1]).scatter_add_(0, Rf_indice.to(torch.int64), Rf_values))
        t2 = time.time()
        # endregion

        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)

        R0 = torch.zeros(self.RGC_list_indexStart[-1])
        # Convert R_indices to int64 explicitly for scatter operation
        R0.scatter_add_(0, R_indices.to(torch.int64), R_values)
        t0 = time.time()
        R = R0
        #region consider the constraints
        for c in self._constraints.values():
            R_new = c.modify_R_K(
                RGC, R0, if_onlyforce=True)
            R = R + R_new
        t4 = time.time()
        #endregion

        # get the global stiffness matrix and force vector

        R = R[self.RGC_remain_index_flatten]

        t6 = time.time()
        return R
    
    def assemble_Stiffness_Matrix(self,
                                   RGC: list[torch.Tensor] = None, GC: torch.Tensor = None):
        """
        Assemble the stiffness matrix.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
            GC (torch.Tensor, optional): The generalized coordinates. If provided, it will be converted to RGC internally. Defaults to None.

        Returns:
            tuple: A tuple containing:
                R (torch.Tensor): The global force vector.
                K_indices (torch.Tensor): The indices of the global stiffness matrix.
                K_values (torch.Tensor): The values of the global stiffness matrix.
        """

        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        #region evaluate the structural K and R
        R0, K_indices, K_values = self._assemble_generalized_Matrix(
            RGC)
        # endregion
        R, K_indices, K_values = self._assemble_reduced_Matrix(
            RGC, R0, K_indices, K_values)

        return R, K_indices, K_values

    def _assemble_generalized_Matrix(self,
                                     RGC: list[torch.Tensor] = None, GC: torch.Tensor = None):
        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        #region evaluate the structural K and R
        t0 = time.time()
        K_values = []
        K_indices = []
        R_values = []
        R_indices = []

        for ins in self._instances.keys():
            Ra_indice, Ra_values, Ka_indice, Ka_value = self._instances[ins].structural_stiffness(
                RGC=RGC)
            K_values.append(Ka_value)
            K_indices.append(Ka_indice)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)
        t1 = time.time()

        ff = []
        for f in self._loads.values():
            Rf_indice, Rf_values, Kf_indice, Kf_value = f.get_stiffness(
                RGC=RGC)
            K_values.append(-Kf_value)
            K_indices.append(Kf_indice)
            R_values.append(-Rf_values)
            R_indices.append(Rf_indice)

            ff.append(torch.zeros(self.RGC_list_indexStart[-1]).scatter_add_(0, Rf_indice.to(torch.int64), Rf_values))
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
        R = R0
        #region consider the constraints
        for c in self._constraints.values():
            R_new, Kc_indices, Kc_values = c.modify_R_K(
                RGC, R0, K_indices, K_values)
            K_indices = torch.cat([K_indices, Kc_indices], dim=1)
            K_values = torch.cat([K_values, Kc_values])
            R = R + R_new
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
                                RGC: list[torch.Tensor] = None, GC: torch.Tensor = None) -> float:
        """
        Calculate the total potential energy of the finite element model.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.

        Returns:
            float: The total potential energy.
        """

        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        # structural energy
        energy = 0
        for ins in self._instances.values():
            energy = energy + ins.potential_energy(RGC=RGC)

        # force potential
        for f in self._loads.values():
            energy = energy - f.get_potential_energy(RGC=RGC)

        return energy
    
    # endregion

    # region for Dynamic Mass Matrix

    def assemble_mass_matrix(self, GC_now: torch.Tensor):
        mass_indices = [self.mass_matrix_indices]
        mass_values = [self.mass_matrix_values]
        RGC = self._GC2RGC(GC_now)
        for c in self._constraints.values():
            indices_now, values_now = c.modify_mass_matrix(mass_indices=self.mass_matrix_indices, mass_values=self.mass_matrix_values, RGC=RGC)
            mass_indices.append(indices_now)
            mass_values.append(values_now)

        mass_indices = torch.cat(mass_indices, dim=1)
        mass_values = torch.cat(mass_values, dim=0)

        # get the global stiffness matrix and force vector
        index_remain = self.RGC_remain_index_flatten[mass_indices[0].cpu(
        )] & self.RGC_remain_index_flatten[mass_indices[1].cpu()]
        mass_values = mass_values[index_remain]
        mass_indices = mass_indices[:, index_remain]
        t44 = time.time()

        mass_indices[0] = mass_indices[0].unique(return_inverse=True)[1]
        mass_indices[1] = mass_indices[1].unique(return_inverse=True)[1]

        return mass_indices, mass_values

    # endregion

    # region GC
    def _allocate_RGC(self, size: list[int] | tuple[int], *args, **kwargs):
        """
        Allocate memory for the RGC data structure.

        Args:
        - size: A list of integers representing the size of the RGC tensor.
        - name: (optional) A string representing the name of the RGC tensor.

        Returns:
        None
        """

        index_now = len(self.RGC)

        self.RGC.append(torch.randn(size) * 0)
        self.RGC_remain_index.append(np.zeros(size, dtype=bool))
        self._RGC_size.append(size)
        self.RGC_list_indexStart.append(
            self.RGC_list_indexStart[-1] + np.prod(size))

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

        for c in self._constraints.values():
            RGC = c.modify_RGC(RGC)

        for b in self._boundarys.values():
            RGC = b.modify_RGC(RGC)

        return RGC

    def _RGC2GC(self, RGC: list[torch.Tensor]):
        GC = torch.cat([
            RGC[i][self.RGC_remain_index[i]].flatten() for i in range(len(RGC))
        ],
                       dim=0)
        return GC

    def refine_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        RGC_out = [RGC[i].clone().detach() for i in range(len(RGC))]
        for instance in self._instances.values():
            RGC_out = instance.refine_RGC(RGC_out)
        return RGC_out

    # endregion

    # region Instance Management

    def add_part(self, part: Part, name: str = None) -> None:
        if name is None:
            name = part.__class__.__name__
            number = len(self._parts)
            while ('%s-%d' % (name, number)) in self._parts:
                number += 1
            name = '%s-%d' % (name, number)

        if name in self._parts:
            raise ValueError(f"Part with name {name} already exists in the assembly.")
        self._parts[name] = part

    def get_part(self, name: str) -> Part:
        if name not in self._parts:
            raise ValueError(f"Part with name {name} does not exist in the assembly.")
        return self._parts[name]
    
    def delete_part(self, name: str) -> None:
        if name in self._parts:
            del self._parts[name]
        else:
            raise ValueError(f"Part with name {name} does not exist in the assembly.")

    def add_instance(self, instance: Instance, name: str = None) -> None:
        if name is None:
            name = instance.__class__.__name__
            number = len(self._instances)
            while ('%s-%d' % (name, number)) in self._instances:
                number += 1
            name = '%s-%d' % (name, number)

        if name in self._instances:
            raise ValueError(f"Instance with name {name} already exists in the assembly.")
        self._instances[name] = instance
        
    def get_instance(self, name: str) -> Instance:
        if name not in self._instances:
            raise ValueError(f"Instance with name {name} does not exist in the assembly.")
        return self._instances[name]

    def delete_instance(self, name: str) -> None:
        if name in self._instances:
            del self._instances[name]
        else:
            raise ValueError(f"Instance with name {name} does not exist in the assembly.")

    def add_reference_point(self, rp: ReferencePoint, name: str = None):
        """
        Adds a reference point to the FEA object.

        Parameters:
            node (torch.Tensor): The node to be added as a reference point.

        Returns:
            str: The name of the reference point.
        """

        if name is None:
            number = len(self._reference_points)
            while ('rp-%d' % number) in self._reference_points:
                number += 1
            name = 'rp-%d' % number

        self._reference_points[name] = rp

        return name

    def get_reference_point(self, name: str) -> ReferencePoint:
        """
        Retrieves a reference point from the FEA object.

        Parameters:
        - name (str): The name of the reference point to be retrieved.

        Returns:
        - ReferencePoint: The requested reference point.
        """
        if name in self._reference_points:
            return self._reference_points[name]
        else:
            raise ValueError(
                f"Reference point '{name}' not found in the model.")

    def delete_reference_point(self, name: str):
        """
        Deletes a reference point from the FEA object.

        Parameters:
        - name (str): The name of the reference point to be deleted.

        Returns:
        - None
        """
        if name in self._reference_points:
            del self._reference_points[name]
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
            name = load.__class__.__name__
            number = len(self._loads)
            while ('%s-%d' % (name, number)) in self._loads:
                number += 1
            name = '%s-%d' % (name, number)
        self._loads[name] = load

        return name
    
    def add_loads(self, loads_dict: dict[str, loads.BaseLoad]):
        """
        Add multiple loads to the FEA model.

        Parameters:
            loads_dict (dict): A dictionary where keys are load names and values are Load.Force_Base objects.

        Returns:
            None
        """
        for name, load in loads_dict.items():
            self.add_load(load, name)

    def get_load(self, name: str) -> loads.BaseLoad:
        """
        Retrieve a load from the FEA model.

        Parameters:
            name (str): The name of the load to be retrieved.

        Returns:
            Load.Force_Base: The requested load.
        """
        if name in self._loads:
            return self._loads[name]
        else:
            raise ValueError(f"Load '{name}' not found in the model.")

    def delete_load(self, name: str):
        """
        Delete a load from the FEA model.

        Parameters:
            name (str): The name of the load to be deleted.

        Returns:
            None
        """
        if name in self._loads:
            del self._loads[name]
        else:
            raise ValueError(f"Load '{name}' not found in the model.")

    def delete_all_loads(self):
        """
        Delete all loads from the FEA model.

        Returns:
            None
        """
        self._loads.clear()

    def get_load_parameters(self) -> dict[str, torch.Tensor]:
        """
        Get parameters about all loads in the FEA model.

        Returns:
            dict: A dictionary where keys are load names and values are numpy arrays containing load parameters.
        """
        load_info = {}
        for name, load in self._loads.items():
            load_info[name] = load._parameters
        return load_info
    
    def set_load_parameters(self, load_info: dict[str, torch.Tensor]):
        """
        Set parameters for loads in the FEA model.

        Args:
            load_info (dict): A dictionary where keys are load names and values are torch tensors containing load parameters.

        Returns:
            None
        """
        for name, info in load_info.items():
            if name in self._loads:
                self._loads[name]._parameters = info
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
            number = len(self._constraints)
            name = constraint.__class__.__name__
            while ('%s-%d' % (name, number)) in self._constraints:
                number += 1
            name = '%s-%d' % (name, number)
        self._constraints[name] = constraint
        return name

    def get_constraint(self, name: str) -> constraints.BaseConstraint:
        """
        Retrieve a constraint from the FEA model.

        Parameters:
            name (str): The name of the constraint to be retrieved.

        Returns:
            Constraints.Constraints_Base: The requested constraint.
        """
        if name in self._constraints:
            return self._constraints[name]
        else:
            raise ValueError(f"Constraint '{name}' not found in the model.")

    def delete_constraint(self, name: str):
        """
        Delete a constraint from the FEA model.

        Parameters:
            name (str): The name of the constraint to be deleted.

        Returns:
            None
        """
        if name in self._constraints:
            del self._constraints[name]
        else:
            raise ValueError(f"Constraint '{name}' not found in the model.")

    # region Boundary Management

    def add_boundary(self, boundary: object, name: str = None):
        """
        Add a boundary condition object to the model.

        Parameters:
            boundary: The boundary condition object (from assemble.boundarys).

        Returns:
            str: The name of the boundary.
        """
        if name is None:
            name = boundary.__class__.__name__
            number = len(self._boundarys)
            while (f"{name}-{number}") in self._boundarys:
                number += 1
            name = f"{name}-{number}"
        self._boundarys[name] = boundary
        return name

    def get_boundary(self, name: str):
        if name in self._boundarys:
            return self._boundarys[name]
        else:
            raise ValueError(f"Boundary '{name}' not found in the model.")

    def delete_boundary(self, name: str):
        if name in self._boundarys:
            del self._boundarys[name]
        else:
            raise ValueError(f"Boundary '{name}' not found in the model.")

    # endregion

    # endregion


#========= Source code for Serializable.BaseSolver =========#
class BaseSolver(Serializable):
    """
    Base class for all solvers in the FEA module.
    """
    def __init__(self) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """

        self.assembly: Assembly = None
        """ The assembly of the finite element model. """

    @property
    def serialized_attributes(self):
        """Get the list of attributes to be serialized."""
        serialized_attrs = super().serialized_attributes
        serialized_attrs = [attr for attr in serialized_attrs if attr != 'assembly']
        return serialized_attrs

    def initialize(self, assembly: Assembly, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            assembly (Assembly): The assembly of the finite element model.
        """
        self.assembly = assembly

    def solve(self, GC0: torch.Tensor = None, *args, **kwargs) -> BaseResult:
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            BaseResult: The result of the finite element analysis. The result object should include convergence metadata (e.g. `converged`).
        """
        pass


#========= Source code for Serializable.BaseSolver.DynamicImplicitSolver =========#
class DynamicImplicitSolver(BaseSolver):

    def __init__(self, maximum_iteration: int = 10000, deltaT: float = 1e-2, time_end: float = 1.0, tol_error: float = 1e-5) -> None:
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

        self.tol_error: float = tol_error
        """
        The tolerance error for the solver.
        """

        self.__low_alpha_count = 0
        
        self._GC_list: list[torch.Tensor] = None
        """ The generalized coordinates of the nodes. """

        self._GV_list: list[torch.Tensor] = None
        """ The velocity of the nodes. """

        self._time_list: list[float] = []
        """ The time of each step. """

        self._deltaT : float = deltaT
        """ The time increment of each step. """

        self._time_end : float = time_end
        """ The end time of the simulation. """

        self._gamma : float = 0.5
        """ The Newmark gamma parameter. """

        self._beta : float = 0.25
        """ The Newmark beta parameter. """

    def initialize(self, assembly: Assembly):
        """
        Initialize the solver with the assembly and initial conditions.
        """
        super().initialize(assembly=assembly)
        self.assembly.initialize_dynamic()

        self._GC_list = []
        self._GV_list = []
        self._GA_list = []
        self._time_list = []
        self._deltaT_list = []

    def set_deltaT(self, deltaT: float):
        """
        Update the time increment for the next step.

        Args:
            deltaT (float): The time increment for the next step.
        """
        self._deltaT = deltaT

    def get_total_energy(self, GC_now: torch.Tensor, GC0: torch.Tensor, GV0: torch.Tensor, GA0: torch.Tensor, deltaT: float = None) -> float:

        if deltaT is None:
            deltaT = self._deltaT

        potential_energy = self.assembly._total_Potential_Energy(GC=GC_now)

        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC_now)
        GV_now = self.get_next_velocity(GC_now=GC_now, GC0=GC0, GV0=GV0, GA0=GA0, deltaT=deltaT)[0]
        kinetic_energy_all = mass_values * GV_now[mass_indices[0]] * GV_now[mass_indices[1]] / 2
        kinetic_energy = kinetic_energy_all.sum()

        return potential_energy + kinetic_energy
    
    def get_incremental_energy(self, GC_now: torch.Tensor, GC_pre: torch.Tensor, deltaT: float = None) -> float:

        if deltaT is None:
            deltaT = self._deltaT

        potential_energy = self.assembly._total_Potential_Energy(GC=GC_now)

        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC_now)
        GC_diff = GC_now - GC_pre
        kinetic_energy_all = mass_values * GC_diff[mass_indices[0]] * GC_diff[mass_indices[1]] / (2 * self._beta * deltaT ** 2)
        kinetic_energy = kinetic_energy_all.sum()

        return potential_energy + kinetic_energy

    def get_incremental_stiffness_matrix(self, GC_now: torch.Tensor, GC_pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the stiffness matrix for the current configuration.

        Args:
            GC_now (torch.Tensor): Current generalized coordinates.
            GC0 (torch.Tensor): Previous generalized coordinates.
            GV0 (torch.Tensor): Previous velocities.
            GA0 (torch.Tensor): Previous accelerations.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Residual force, Indices and values of the stiffness matrix.
        """
        # assemble the internal force and stiffness matrix
        Rv, Kv_indices, Kv_values = self.assembly.assemble_Stiffness_Matrix(GC=GC_now)
        
        # assemble the mass matrix
        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC_now)
        GC_diff = GC_now - GC_pre
        Ri_values = mass_values * GC_diff[mass_indices[0]] / (self._beta * self._deltaT ** 2)
        Ri_indices = mass_indices[1]
        Ri = torch.zeros_like(GC_now).scatter_add_(0, Ri_indices, Ri_values)
        Ki_values = mass_values / (self._beta * self._deltaT ** 2)
        Ki_indices = mass_indices

        K_indices = torch.cat([Kv_indices, Ki_indices], dim=1)
        K_values = torch.cat([Kv_values, Ki_values], dim=0)
        R = Rv + Ri

        return R, K_indices, K_values

    def solve(self, GC0: torch.Tensor = None, GV0: torch.Tensor = None, *args, **kwargs) -> bool:
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

        if GC0 is None:
            GC0 = self.assembly.GC.clone()

        if GV0 is None:
            GV0 = torch.zeros_like(self.assembly.GC)

        self._GC_list = [GC0]
        self._GV_list = [GV0]
        self._GA_list = [self.get_current_acceleration(GC0=GC0, GV0=GV0)]
        self._time_list = [0.0]
        E_history = [self.get_total_energy(GC_now=GC0, GC0=GC0, GV0=GV0, GA0=self._GA_list[-1])]
        
        # start the iteration
        iteration = 0
        while True:
            print('---' * 8, 'FEA Step %d' % (iteration + 1), '---' * 8)
            print('time:%.8f, deltaT:%.8f' % (self._time_list[-1], self._deltaT))
            t0 = time.time()

            GC_pre = self._GC_list[-1] + self._deltaT * self._GV_list[-1] + (self._deltaT ** 2) * self._GA_list[-1] * (1 - 2 * self._beta) / 2

            GC_now = self._solve_iteration(GC0=self._GC_list[-1],
                                            GC_pre=GC_pre,
                                            deltaT=self._deltaT,
                                            tol_error=self.tol_error)
            t2 = time.time()

            # print the information
            print('total_iter:%d, total_time:%.2f' % (iteration, t2 - t0))
            E = self.get_total_energy(GC_now=GC_now, GC0=self._GC_list[-1], GV0=self._GV_list[-1], GA0=self._GA_list[-1])

            # energy_diff = (E - E_history[-1]).abs() / abs(E_history[-1])
            # if energy_diff > 1e-2:
            #     print('energy increase too much, reduce the time step')
            #     self._deltaT = self._deltaT / 2
            #     print('new deltaT:%.8f' % self._deltaT)
            #     print('---' * 8, 'FEA Continued', '---' * 8, '\n')
            #     continue
            # elif energy_diff < 1e-3:
            #     self._deltaT = self._deltaT * 1.2

            print('max_error:%.4e' % (((E - E_history[-1]) / E_history[-1]).abs()))
            print('---' * 8, 'FEA Continued', '---' * 8, '\n')

            # update the results
            GVnew, GAnew = self.get_next_velocity(GC_now=GC_now, GC0=self._GC_list[-1], GV0=self._GV_list[-1], GA0=self._GA_list[-1], deltaT=self._deltaT)
            E_history.append(E)
            self._GC_list.append(GC_now)
            self._GV_list.append(GVnew)
            self._GA_list.append(GAnew)
            self._time_list.append(self._time_list[-1] + self._deltaT)

            iteration += 1
            if self._time_list[-1] >= self._time_end:
                break
        
        self.GC=self._GC_list[-1]
        self.assembly.RGC = self.assembly.refine_RGC(self.assembly._GC2RGC(self.assembly.GC))
        t2 = time.time()

        # print the information
        print('total_time:%.2f' % (t2 - t0))
        R = self.assembly.assemble_Stiffness_Matrix(RGC=self.assembly.RGC)[0]
        print('max_error:%.4e' % (R.abs().max()))
        print('---' * 8, 'FEA Finished', '---' * 8, '\n')

        result = DynamicResult(
            GC_list=self._GC_list,
            GV_list=self._GV_list,
            GA_list=self._GA_list,
            time_list=self._time_list,
            load_params=self.assembly.get_load_parameters(),
            total_time=t2 - t0
        )

        return result

    def get_next_velocity(self, GC_now: torch.Tensor, GC0: torch.Tensor, GV0: torch.Tensor, GA0: torch.Tensor, deltaT: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the velocity and acceleration based on the Newmark-beta method.

        Args:
            GC_now (torch.Tensor): Current generalized coordinates.
            GC0 (torch.Tensor): Previous generalized coordinates.
            GV0 (torch.Tensor): Previous velocities.
            GA0 (torch.Tensor): Previous accelerations.
            deltaT (float): Time increment.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Current velocities and accelerations.
        """
        # Newmark-beta method for velocity and acceleration update

        GV_now = self._gamma / (self._beta * deltaT) * (GC_now - GC0) + (1 - self._gamma / self._beta) * GV0 + deltaT * (1 - self._gamma / (2 * self._beta)) * GA0
        GA_now = 1 / (self._beta * deltaT ** 2) * (GC_now - GC0) - 1 / (self._beta * deltaT) * GV0 - (1 / (2 * self._beta) - 1) * GA0

        return GV_now, GA_now

    def get_current_acceleration(self, GC0: torch.Tensor, GV0: torch.Tensor) -> torch.Tensor:
        """
        Calculate the current acceleration based on the newton's law.

        Args:
            GC0 (torch.Tensor): Previous generalized coordinates.
            GV0 (torch.Tensor): Previous velocities.
            GA0 (torch.Tensor): Previous accelerations.

        Returns:
            torch.Tensor: Current accelerations.
        """
        Rv = self.assembly.assemble_Stiffness_Matrix(GC=GC0)[0]

        # Newmark initial acceleration
        # M * GA0 = F_ext(GC0) - F_int(GC0)
        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC0)
        GA0 = _linear_solver.pypardiso_solver(mass_indices, mass_values, Rv)
        return GA0
    
    # region solve iteration

    def _line_search(self,
                     GC_now: torch.Tensor,
                     GC_pre: torch.Tensor,
                     dGC: torch.Tensor,
                     R: torch.Tensor,
                     energy0: float, deltaT: float, *args, **kwargs):
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
            GCnew = GC_now + alpha * dGC
            # GCnew.requires_grad_()
            energy_new = self.get_incremental_energy(GC_now=GCnew, GC_pre=GC_pre, deltaT=deltaT)

            if torch.isnan(energy_new) or torch.isinf(
                    energy_new) or \
                energy_new > energy0 + c1 * deltaE * alpha or \
                (alpha * dGC).abs().max() > self._maximum_step_length:
                alpha = 0.5 * alpha
                if alpha < 1e-12:
                    alpha = 0.0
                    GCnew = GC_now.clone()
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
        return alpha, GCnew.detach(), energy_new

    def _solve_iteration(self,
                         GC0: torch.Tensor,
                         GC_pre: torch.Tensor,
                         deltaT: float,
                         tol_error: float):

        GC_now = GC0.clone()

        # iteration now
        self._iter_now = 0

        # initialize the time
        t00 = time.time()

        # initialize the energy
        energy = [
            self.get_incremental_energy(GC_now=GC_now, GC_pre=GC_pre, deltaT=deltaT)
        ]

        dGC = torch.zeros_like(GC0)

        # record the number of low alpha
        low_alpha = 0
        alpha = 0

        # begin the iteration
        while True:

            if self._iter_now > self.maximum_iteration:
                print('maximum iteration reached')
                self.assembly.GC = GC_now
                return False

            # calculate the force vector and tangential stiffness matrix
            t1 = time.time()
            R, K_indices, K_values = self.get_incremental_stiffness_matrix(
                GC_now=GC_now, GC_pre=GC_pre)

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
                    GC_now=GC_now, GC_pre=GC_pre, dGC=dGC, R=R, energy0=energy[-1], deltaT=deltaT)

            if alpha==0 and R.abs().max() > tol_error:
                self.assembly.GC = GC_now
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
                    self.assembly.GC = GC_now
                    break
                return False



            # self.show_surface(nodes=self.nodes+RGC[0])

            # update the energy
            energynew = self.get_incremental_energy(GC_now=GCnew, GC_pre=GC_pre, deltaT=deltaT)
            energy.append(energynew)

            # update the GC
            GC_now = GCnew

            t4 = time.time()

            # return the index to the first line
            if self._iter_now > 1:
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

        return GC_now

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


#========= Source code for Serializable.BaseSolver.DynamicExplicitSolver =========#
class DynamicExplicitSolver(BaseSolver):

    def __init__(self, time_end: float = 1.0, time_per_storage: float = 1e-4) -> None:
        """
        Initialize the Explicit Dynamic Solver.

        Args:
            time_end (float): The end time of the simulation.
            dump_factor (float): Damping factor for numerical stability.
        """
        self._time_end: float = time_end
        self._time_per_storage: float = time_per_storage
        self._next_storage_time: float = 0.0  # 下一个需要存储的时间点


        self._GC_list: list[torch.Tensor] = None
        self._GV_list: list[torch.Tensor] = None
        self._GA_list: list[torch.Tensor] = None
        self._time_list: list[float] = []
        self._deltaT: float = 0.0 # 将在初始化时计算

    def initialize(self, assembly: Assembly):
        """
        Initialize the solver with the assembly and compute critical time step.
        """
        super().initialize(assembly=assembly)
        self.assembly.initialize_dynamic()

        # 1. 计算临界时间步长 (CFL Condition)
        # 这是一个简化估算，精确计算需要遍历所有单元
        # Δt_crit = L_min / c,  c = sqrt(E/ρ)
        # 这里我们用一个经验值或一个估算函数
        self._deltaT = self.estimate_critical_timestep()
        print(f"Estimated critical timestep: {self._deltaT:.4e} s")
        if self._deltaT <= 0:
            raise ValueError("Critical timestep must be positive.")

        print(f"Critical timestep estimated: {self._deltaT:.4e} s")

        self._GC_list = []
        self._GV_list = []
        self._GA_list = []
        self._time_list = []

    def estimate_critical_timestep(self, safety_factor=0.8) -> float:
        """
        Estimates the critical timestep for stability (CFL condition).
        This is a placeholder and should be implemented based on element sizes and material properties.
        """
        # 这是一个非常粗略的估计，您需要根据您的模型进行调整
        # 例如，遍历所有单元，找到最小的 L/c
        # L: 单元特征长度, c: 材料波速 sqrt(E/rho)
        # 假设一个经验值
        estimated_crit_dt = 5e-6 
        return safety_factor * estimated_crit_dt

    def solve(self, GC0: torch.Tensor = None, GV0: torch.Tensor = None, *args, **kwargs) -> bool:
        """
        Solves the finite element analysis problem using the explicit central difference method.
        """
        t_start = time.time()

        # 1. 设置初始条件
        if GC0 is None:
            GC0 = self.assembly.GC.clone()
        if GV0 is None:
            GV0 = torch.zeros_like(self.assembly.GC)

        # 2. 计算初始加速度 a_0 = M⁻¹ * (F_ext(0) - F_int(u_0))
        # F_int(u_0) 通常为0，除非有预应力
        R0 = -self.assembly.assemble_force(GC=GC0) # 只取残余力向量
        mass_inv = self.get_lumped_mass_inv(GC0)
        GA0 = mass_inv * R0

        # 3. 初始化列表并存储初始状态
        self._GC_list.append(GC0)
        self._GV_list.append(GV0)
        self._GA_list.append(GA0)
        self._time_list.append(0.0)

        # 4. 计算 "半步" 初始速度 v_{-1/2}
        GV_half_prev = GV0 - 0.5 * self._deltaT * GA0

        # 5. 主时间步循环
        iteration = 0
        current_time = 0.0
        GA_now = GA0
        while current_time < self._time_end:
            
            # a. 获取上一步的状态
            GC_prev = self._GC_list[-1]
            
            # b. 更新 "半步" 速度: v_{n+1/2} = v_{n-1/2} + Δt * a_n
            GV_half_now = GV_half_prev + self._deltaT * GA_now

            # c. 更新位移: u_{n+1} = u_n + Δt * v_{n+1/2}
            GC_now = GC_prev + self._deltaT * GV_half_now

            # d. 计算新的内力和外力: R_{n+1} = F_ext(t_{n+1}) - F_int(u_{n+1})
            # 注意：assemble_Stiffness_Matrix 返回的是 F_ext - F_int
            R_now = -self.assembly.assemble_force(GC=GC_now)

            # e. 计算新的加速度: a_{n+1} = M⁻¹ * R_{n+1}
            
            GA_now = mass_inv * R_now

            # f. 更新节点速度 (用于输出): v_{n+1} = (v_{n+1/2} + v_{n-1/2}) / 2
            GV_now = 0.5 * (GV_half_now + GV_half_prev)

            # g. 更新时间
            current_time += self._deltaT

            # h. 【新增】检查是否需要存储当前步结果
            if current_time >= self._next_storage_time or current_time >= self._time_end:
                self._GC_list.append(GC_now)
                self._GV_list.append(GV_now)
                self._GA_list.append(GA_now)
                self._time_list.append(current_time)
                
                # 更新下一个存储时间点
                self._next_storage_time += self._time_per_storage

            # i. 准备下一次迭代
            GV_half_prev = GV_half_now
            iteration += 1

            if iteration % 100 == 0: # 每100步打印一次信息
                print(f"Step: {iteration}, Time: {current_time:.4e} s, Max Disp: {GC_now.abs().max():.4e}, time_cost: {time.time() - t_start:.2f} s")

        # 6. 结束
        self.GC = self._GC_list[-1]
        self.assembly.RGC = self.assembly.refine_RGC(self.assembly._GC2RGC(self.assembly.GC))
        t_end = time.time()
        print('---' * 8, 'Explicit FEA Finished', '---' * 8)
        print(f'Total steps: {iteration}, Total time: {t_end - t_start:.2f} s')

        result = DynamicResult(
            GC_list=self._GC_list,
            GV_list=self._GV_list,
            GA_list=self._GA_list,
            time_list=self._time_list,
            load_params=self.assembly.get_load_parameters(),
            total_time=t_end - t_start
        )
        return result

    def get_lumped_mass_inv(self, GC: torch.Tensor) -> torch.Tensor:
        """
        Assembles the lumped mass matrix and returns its inverse.
        For a diagonal matrix, the inverse is just the reciprocal of its diagonal elements.
        """
        mass_indices, mass_values = self.assembly.assemble_mass_matrix(GC_now=GC)
        
        # 创建一个完整的质量向量 (对角线)
        mass_vector = torch.zeros(GC.shape[0], device=GC.device, dtype=GC.dtype)
        
        # 使用 scatter_add_ 来集成所有质量项到对角线上
        # 注意：这假设 assemble_mass_matrix 返回的是完整的 M_ij 矩阵项
        # 如果它只返回上三角或下三角，需要相应调整
        # 这里我们假设它返回了所有项，包括对角线 M_ii
        mass_vector.scatter_add_(0, mass_indices[0], mass_values * (mass_indices[0] == mass_indices[1]))
        mass_vector.scatter_add_(0, mass_indices[1], mass_values * (mass_indices[0] != mass_indices[1]))

        # 防止除以零
        mass_vector[mass_vector.abs() < 1e-12] = 1.0
        
        return 1.0 / mass_vector


#========= Source code for Serializable.BaseSolver.StaticImplicitSolver =========#
class StaticImplicitSolver(BaseSolver):

    def __init__(self, maximum_iteration: int = 10000, tol_error: float = 1e-5) -> None:
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

        self.tol_error: float = tol_error
        """
        The tolerance error for the solver.
        """

        self.__low_alpha_count = 0


    def solve(self, GC0: torch.Tensor = None, need_jacobian: bool = False, *args, **kwargs) -> StaticResult:
        """
        Solves the finite element analysis problem and returns a StaticResult object.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            StaticResult: The result object containing GC, jacobian (optional), and convergence status.
        """
        # initialize the RGC
        t0 = time.time()
        # start the iteration
        if GC0 is None:
            GC0 = self.assembly.GC
        with torch.no_grad():
            solve_output = self._solve_iteration(GC=GC0, tol_error=self.tol_error)

        GC_final, total_time_iter, time_items, converged = solve_output
        self.assembly.GC = GC_final
        self.assembly.RGC = self.assembly.refine_RGC(self.assembly._GC2RGC(GC_final))
        t2 = time.time()

        # print the information
        print('total_iter:%d, total_time:%.2f' % (self._iter_now, t2 - t0))
        R = self.get_stiffness_matrix(GC_now=GC_final)[0]
        print('max_error:%.4e' % (R.abs().max()))
        print('---' * 8, 'FEA Finished', '---' * 8, '\n')

        # build the result object
        fe_result = StaticResult(
            GC=GC_final,
            load_params=self.assembly.get_load_parameters(),
            total_time=total_time_iter,
            time_items=time_items,
            converged=converged,
        )

        if need_jacobian:
            jacobian = self.get_jacobian(result=fe_result)
            fe_result.jacobian = jacobian

        return fe_result
   
    def get_jacobian(self, result: StaticResult, load_names: list[str] = None) -> torch.Tensor:
        """
        Calculate the Jacobian matrix for the current configuration.

        Args:
            result (StaticResult): The result object containing the current state.
            load_names (list[str], optional): The names of the loads for which to compute the Jacobian. If None, all loads are considered.
        Returns:
            torch.Tensor: The Jacobian matrix.
                shape: (num_dofs, num_load_params).
        """
        # set the load parameters to the assembly
        self.assembly.set_load_parameters(result.load_params)

        # if load_names is None, compute for all loads
        if load_names is None:
            load_names = list(self.assembly._loads.keys())
        
        # get the current load parameters as a single tensor
        total_params_list = []
        for load_name in load_names:
            load = self.assembly._loads[load_name]
            total_params_list.append(load._parameters.flatten())
        total_params = torch.cat(total_params_list, dim=0)

        # define the closure function to compute R
        def closure_R(total_params: torch.Tensor):

            index_now = 0
            for load_name in load_names:
                load = self.assembly._loads[load_name]
                param_len = load._parameters.numel()
                load._parameters = total_params[index_now:index_now+param_len].reshape(load._parameters.shape)
                index_now += param_len
            R = self.assembly.assemble_force(GC=result.GC.to(self.assembly.device))

            # remove the leaf parameters
            for load_name in load_names:
                load = self.assembly._loads[load_name]
                load._parameters = load._parameters.detach()

            return R
        

        from torch.autograd.functional import jvp
        
        num_params = total_params.numel()
        if num_params > 0:
            Rdp_cols = []

            for i in range(num_params):
                # Compute Jacobian-Vector Product for the i-th parameter
                # This computes the i-th column of the Jacobian
                basis_vector_now = torch.zeros_like(total_params)
                basis_vector_now[i] = 1.0
                _, col = jvp(closure_R, total_params, v=basis_vector_now, create_graph=False)
                Rdp_cols.append(col)
            
            Rdp = torch.stack(Rdp_cols, dim=1)
        else:
            # Handle case with no parameters
            R_dummy = closure_R(total_params)
            Rdp = torch.zeros((R_dummy.shape[0], 0), device=total_params.device, dtype=total_params.dtype)


        if result.if_factorized is False:
            result.factorize_stiffness_matrix(assembly=self.assembly)
        
        jacobian = -result.K_solver.solve(result.K_sp, Rdp.cpu().numpy())
        jacobian = jacobian.reshape(-1, num_params) # Shape: (num_dofs, num_load_params)

        jacobian_output = {}
        index_now = 0
        for load_name in load_names:
            load = self.assembly._loads[load_name]
            param_len = load._parameters.numel()
            jacobian_output[load_name] = torch.from_numpy(jacobian[:, index_now:index_now+param_len]).to(result.GC.dtype).to(result.GC.device)
            index_now += param_len

        return jacobian_output
    
    def get_total_energy(self, GC_now: torch.Tensor) -> float:
        
        potential_energy = self.assembly._total_Potential_Energy(GC=GC_now)
        return potential_energy
    
    def get_stiffness_matrix(self, GC_now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the stiffness matrix for the current configuration.

        Args:
            GC_now (torch.Tensor): Current generalized coordinates.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Indices and values of the stiffness matrix.
        """
        R, K_indices, K_values = self.assembly.assemble_Stiffness_Matrix(GC=GC_now)
        return R,K_indices, K_values


    # region sensitivity analysis
    def get_sensitivity(
        self,
        fe_result: StaticResult,
        design_vars: torch.Tensor,
        apply_func: Callable[[Assembly, torch.Tensor], None],
        compute_objective_func: Callable[[StaticResult, Assembly], torch.Tensor]
    ) -> torch.Tensor:
        """
        Core functional implementation of the adjoint sensitivity analysis.

        This function calculates the gradient of an objective function with respect to design variables,
        constrained by the finite element equilibrium equations, using the discrete adjoint method.

        Args:
            fe_result (StaticResult): The solution containing the factorized stiffness matrix (K) and
                displacement/generalized coordinates (GC). K should be pre-factorized for efficiency.
            assembly (Assembly): The Finite Element Assembly object containing parts, elements, loads, etc.
            design_vars (torch.Tensor): A tensor representing the design variables.
                It must be the source of gradients for `apply_func`.
            apply_func (Callable[[Assembly, torch.Tensor], None]): 
                A callback to apply design variables to the assembly.
                - Signature: `def apply_func(assembly: Assembly, design_vars: torch.Tensor) -> None`
                - Behavior: Modify `assembly` in-place using `design_vars`. Operations must be traceable
                by Autograd (e.g., `part.nodes = original_nodes + design_vars.reshape_as(part.nodes)`).
            compute_objective_func (Callable[[StaticResult, Assembly], torch.Tensor]): 
                A callback to compute the objective scalar.
                - Signature: `def compute_objective_func(fe_result: StaticResult, assembly: Assembly) -> torch.Tensor`
                - Args: `fe_result` contains the necessary solution data.
                - Returns: A scalar tensor representing the objective value (e.g., compliance, stress).

        Returns:
            torch.Tensor: The gradient of the objective with respect to `design_vars`.
                Shape matches `design_vars`.
        """
        try:
            # 0. Set load parameters
            self.assembly.set_load_parameters(fe_result.load_params)

            # 1. Factorize system if needed
            if fe_result.if_factorized is False:
                fe_result.factorize_stiffness_matrix(assembly=self.assembly)

            # 2. Prepare Autograd graph
            design_vars_grad = design_vars.clone().detach().requires_grad_(True)
            GC_grad = fe_result.GC.clone().detach().requires_grad_(True)
            fe_result.GC = GC_grad  # Use GC_grad for the assembly to track gradients through R and K

            # 3. Apply Design Variables
            apply_func(self.assembly, design_vars_grad)
            self.assembly.initialize()

            # 4. Compute Objective
            objective = compute_objective_func(fe_result, self.assembly)

            # 5. Backward Pass 1 (d_Obj / d_Vars and d_Obj / d_U)
            objective.backward(retain_graph=True)

            # 6. Adjoint Equation Solve (K * lambda = - d_Obj/d_U)
            if GC_grad.grad is None:
                raise RuntimeError("Objective function must depend on the displacement GC.")
                
            LdU = GC_grad.grad.clone().detach()
            W = fe_result.K_solver.solve(fe_result.K_sp, -LdU.cpu().numpy())
            W_tensor = torch.tensor(W, dtype=GC_grad.dtype, device=GC_grad.device)

            # 7. Backward Pass 2 (Total Sensitivity, lambda^T * R)
            R = self.assembly.assemble_force(GC=fe_result.GC)
            work = torch.dot(W_tensor, R)
            work.backward()
            
            if design_vars_grad.grad is None:
                return torch.zeros_like(design_vars)
        
        finally:
            # Cleanup: Detach all tensors in assembly to prevent graph explosion in next run
            self._detach_recursive(self.assembly)
            self._detach_recursive(fe_result)
            fe_result.remove_stored_factorization()


        return design_vars_grad.grad.clone().detach()
    
    def get_jacobian_sensitivity(
        self,
        fe_result: StaticResult,
        design_vars: torch.Tensor,
        load_names: list[str],
        apply_func: Callable[[Assembly, torch.Tensor], None] ,
        compute_objective_func: Callable[[StaticResult, Assembly], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the Jacobian sensitivity (dR/dVars) for the static problem.

        This function computes the sensitivity of the residual forces with respect to design variables,
        which is essential for gradient-based optimization and design.

        Args:
            fe_result (StaticResult): The result object containing the current state and factorized stiffness matrix.
            design_vars (torch.Tensor): A tensor representing the design variables.
                It must be the source of gradients for `apply_func`.
            load_names (list[str]): The names of the loads for which to compute the Jacobian sensitivity.
            apply_func (Callable[[Assembly, torch.Tensor, None]): 
                A callback to apply design variables to the assembly.
                - Signature: `def apply_func(assembly: Assembly, design_vars: torch.Tensor) -> None`
                - Behavior: Modify `assembly` in-place using `design_vars`. Operations must be traceable
                by Autograd (e.g., `part.nodes = original_nodes + design_vars.reshape_as(part.nodes)`).
            compute_objective_func (Callable[[StaticResult, Assembly], torch.Tensor]): 
                A callback to compute the objective scalar.
                - Signature: `def compute_objective_func(fe_result: StaticResult, assembly: Assembly) -> torch.Tensor`
                - Args: `fe_result` contains the necessary solution data.
                - Returns: A scalar tensor representing the objective value (e.g., compliance, stress).
        Returns:
            torch.Tensor: The sensitivity of the objective with respect to design variables.
                Shape matches `design_vars`.
        """
        try:
            # 0. Set load parameters and prepare jacobian
            self.assembly.set_load_parameters(fe_result.load_params)

            # 1. Factorize system if needed
            if fe_result.if_factorized is False:
                fe_result.factorize_stiffness_matrix(assembly=self.assembly)

            # 2. Prepare Jacobian
            if fe_result.jacobian.keys() >= set(load_names):
                jacobian_dict = fe_result.jacobian
            else:                
                jacobian_dict = self.get_jacobian(result=fe_result, load_names=load_names)
                fe_result.jacobian = jacobian_dict

            if len(jacobian_dict.keys()) == 0:
                jacobian = torch.zeros((fe_result.GC.numel(), 0), device=fe_result.GC.device, dtype=fe_result.GC.dtype)
            else:
                jacobian = torch.cat([jacobian_dict[load_name] for load_name in load_names], dim=1).detach() # Shape: (num_dofs, num_load_params)

            # 3. Prepare Autograd graph
            design_vars_grad = design_vars.clone().detach().requires_grad_(True)
            GC_grad = fe_result.GC.clone().detach().requires_grad_(True)
            jacobian_grad = {k: v.detach().clone().requires_grad_(True) for k, v in jacobian_dict.items()}
            fe_result.GC = GC_grad  # Use GC_grad for the assembly to track gradients through R and K
            fe_result.jacobian = jacobian_grad  # Use jacobian_grad to track gradients through the Jacobian
            
            # 4. Apply Design Variables
            apply_func(self.assembly, design_vars_grad)
            self.assembly.initialize()
            R_grad = self.assembly.assemble_force(GC=fe_result.GC)

            # 5. Compute first adjoint vector W0
            objective = compute_objective_func(fe_result, self.assembly)

            # 6. Get Ldx and Ldy for the adjoint solve
            objective.backward(retain_graph=True)
            if GC_grad.grad is None:
                Ldx = torch.zeros_like(GC_grad)
            else:
                Ldx = GC_grad.grad.clone().detach()

            if len(load_names) > 0:
                Ldy_dict = {k: v.grad.clone().detach() if v.grad is not None else torch.zeros_like(v) for k, v in jacobian_grad.items()}
                Ldy = torch.cat([Ldy_dict[load_name] for load_name in load_names], dim=1).detach() # Shape: (num_dofs, num_load_params)
            else:
                Ldy = torch.zeros((GC_grad.numel(), 0), device=GC_grad.device, dtype=GC_grad.dtype)

            # 7. sensitivity for GC
            W0 = -fe_result.K_solver.solve(fe_result.K_sp, Ldx.cpu().numpy())
            W0_tensor = torch.tensor(W0, dtype=GC_grad.dtype, device=GC_grad.device)

            # 8. For each parameter, compute the Jacobian sensitivity using the chain rule:

            ## get the current load parameters as a single tensor
            total_load_params_list = []
            for load_name in load_names:
                load = self.assembly._loads[load_name]
                total_load_params_list.append(load._parameters.flatten())

            if len(total_load_params_list) > 0:
                total_load_params = torch.cat(total_load_params_list, dim=0)
            else:
                total_load_params = torch.zeros((0,), device=GC_grad.device, dtype=GC_grad.dtype)
            num_load_params = total_load_params.numel()

            obj_part_y = torch.zeros(1, device=GC_grad.device, dtype=GC_grad.dtype)
            K_indices = self.assembly.assemble_Stiffness_Matrix(GC=fe_result.GC)[1]
            wKdp = torch.zeros_like(fe_result.GC)
            for para_idx in range(num_load_params):
                ### compute the Jacobian sensitivity using the chain rule:
                Ldy_now = Ldy[:, para_idx]
                if Ldy_now.abs().sum() < 1e-8:
                    continue


                W1 = fe_result.K_solver.solve(fe_result.K_sp, Ldy_now.cpu().numpy())
                W1_tensor = torch.tensor(W1, dtype=GC_grad.dtype, device=GC_grad.device)
                ### evaluate the stiffness matrix sensitivity dK/dPara using autograd
                def get_Kdp(load_now: torch.Tensor):
                    GC_now = fe_result.GC + jacobian[:, para_idx] * (load_now - load_now.detach())
                    index_now = 0
                    for load_name in load_names:
                        load = self.assembly._loads[load_name]
                        param_len = load._parameters.numel()
                        idx_now = para_idx - index_now
                        if 0 <= idx_now < param_len:
                            load._parameters[idx_now] = load_now
                            break
                        index_now += param_len
                    K_values = self.assembly.assemble_Stiffness_Matrix(GC=GC_now)[2]
                    return K_values
                Kdp_values = torch.autograd.functional.jvp(
                    get_Kdp, total_load_params[para_idx: para_idx+1], torch.ones([1]), create_graph=False)[1]
                Kdp = torch.sparse_coo_tensor(K_indices, Kdp_values, size=fe_result.K_sp.shape)

                wKdp += Kdp@ W1_tensor

                ### evaluate the stiffness matrix sensitivity dR/dPara using autograd
                def get_Rdp(load_now: torch.Tensor):
                    GC_now = GC_grad + jacobian[:, para_idx] * (load_now - load_now.detach())
                    index_now = 0
                    for load_name in load_names:
                        load = self.assembly._loads[load_name]
                        param_len = load._parameters.numel()
                        idx_now = para_idx - index_now
                        if 0 <= idx_now < param_len:
                            load._parameters[idx_now] = load_now
                            break
                        index_now += param_len
                    R = self.assembly.assemble_force(GC=GC_now)
                    return (R * W1_tensor).sum()

                obj_part_y -= torch.autograd.functional.jacobian(get_Rdp, total_load_params[para_idx: para_idx+1], create_graph=True)

            if wKdp.abs().sum() > 1e-8:
                wKdpKinv = fe_result.K_solver.solve(fe_result.K_sp, wKdp.cpu().numpy())
            else:
                wKdpKinv = torch.zeros_like(wKdp)
            wKdpKinv_tensor = torch.tensor(wKdpKinv, dtype=GC_grad.dtype, device=GC_grad.device)
            
            obj_part_x = ((W0_tensor + wKdpKinv_tensor) * R_grad).sum()

            obj_total = obj_part_x + obj_part_y
            obj_total.backward()
            sensitivity = design_vars_grad.grad.clone().detach()

        finally:            
            # Cleanup: Detach all tensors in assembly to prevent graph explosion in next run
            self._detach_recursive(self.assembly)
            self._detach_recursive(fe_result)
            fe_result.remove_stored_factorization()

        return sensitivity
    
    def get_jacobian_sensitivity_multistep(        
        self,
        fe_results: list[StaticResult],
        design_vars: torch.Tensor,
        load_names: list[str],
        apply_func: Callable[[Assembly, torch.Tensor], None] ,
        compute_objective_funcs: Callable[[list[StaticResult], Assembly], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the Jacobian sensitivity (dR/dVars) for the static problem.

        This function computes the sensitivity of the residual forces with respect to design variables,
        which is essential for gradient-based optimization and design.

        Args:
            fe_results (list[StaticResult]): A list of result objects containing the current state and factorized stiffness matrices.
            design_vars (torch.Tensor): A tensor representing the design variables.
                It must be the source of gradients for `apply_func`.
            load_names (list[str]): The names of the loads for which to compute the Jacobian sensitivity.
            apply_func (Callable[[Assembly, torch.Tensor, None]): 
                A callback to apply design variables to the assembly.
                - Signature: `def apply_func(assembly: Assembly, design_vars: torch.Tensor) -> None`
                - Behavior: Modify `assembly` in-place using `design_vars`. Operations must be traceable
                by Autograd (e.g., `part.nodes = original_nodes + design_vars.reshape_as(part.nodes)`).
            compute_objective_funcs (Callable[[list[StaticResult], Assembly], torch.Tensor]): 
                A callback to compute the objective scalar.
                - Signature: `def compute_objective_func(fe_results: list[StaticResult], assembly: Assembly) -> torch.Tensor`
                - Args: `fe_results` contains the necessary solution data.
                - Returns: A scalar tensor representing the objective value (e.g., compliance, stress).
        Returns:
            torch.Tensor: The sensitivity of the objective with respect to design variables.
                Shape matches `design_vars`.
        """
        try:
            design_vars_grad = design_vars.clone().detach().requires_grad_(True)
            apply_func(self.assembly, design_vars_grad)
            self.assembly.initialize()

            # 0. Set gradient required for each step's GC and Jacobians
            for idx, fe_result in enumerate(fe_results):
                # 0. Set load parameters and prepare jacobian
                self.assembly.set_load_parameters(fe_result.load_params)

                # 1. Factorize system if needed
                if fe_result.if_factorized is False:
                    fe_result.factorize_stiffness_matrix(assembly=self.assembly)

                # 2. Prepare Jacobian
                if fe_result.jacobian.keys() >= set(load_names):
                    jacobian_dict = fe_result.jacobian
                else:                
                    jacobian_dict = self.get_jacobian(result=fe_result, load_names=load_names)

                # 3. Prepare Autograd graph
                GC_grad = fe_result.GC.clone().detach().requires_grad_(True)
                jacobian_grad = {k: v.detach().clone().requires_grad_(True) for k, v in jacobian_dict.items()}

                fe_result.GC = GC_grad  # Use GC_grad for the assembly to track gradients through R and K
                fe_result.jacobian = jacobian_grad  # Use jacobian_grad to track gradients through the Jacobian

            # 4. Compute first adjoint vector W0
            objective = compute_objective_funcs(fe_results, self.assembly)

            # 5. Get Ldx and Ldy for the adjoint solve
            objective.backward(retain_graph=True)

            print("Computing Jacobian sensitivity for each step:")
            print(f" -Step 0/{len(fe_results)} sensitivity computed.\r", end='')
            for idx, fe_result in enumerate(fe_results):
                # Each step must use its own load state when assembling R, dR/dp and dK/dp.
                self.assembly.set_load_parameters(fe_result.load_params)
                
                # 6. Get R for the current step
                R_grad = self.assembly.assemble_force(GC=fe_result.GC)

                if len(load_names) > 0:
                    jacobian = torch.cat([fe_result.jacobian[load_name] for load_name in load_names], dim=1).detach()
                else:
                    jacobian = torch.zeros((fe_result.GC.numel(), 0), device=fe_result.GC.device, dtype=fe_result.GC.dtype)

                # 7. Compute Ldx and Ldy for the adjoint solve
                if fe_result.GC.grad is None:
                    Ldx = torch.zeros_like(fe_result.GC)
                else:
                    Ldx = fe_result.GC.grad.clone().detach()

                if len(load_names) > 0:
                    Ldy_dict = {k: v.grad.clone().detach() if v.grad is not None else torch.zeros_like(v) for k, v in fe_result.jacobian.items()}
                    Ldy = torch.cat([Ldy_dict[load_name] for load_name in load_names], dim=1).detach() # Shape: (num_dofs, num_load_params)
                else:
                    Ldy = torch.zeros((fe_result.GC.numel(), 0), device=fe_result.GC.device, dtype=fe_result.GC.dtype)

                # 8. sensitivity for GC
                W0 = -fe_result.K_solver.solve(fe_result.K_sp, Ldx.cpu().numpy())
                W0_tensor = torch.tensor(W0, dtype=fe_result.GC.dtype, device=fe_result.GC.device)

                # 9. For each parameter, compute the Jacobian sensitivity using the chain rule:

                ## get the current load parameters as a single tensor
                total_load_params_list = []
                for load_name in load_names:
                    load = self.assembly._loads[load_name]
                    total_load_params_list.append(load._parameters.flatten())

                if len(total_load_params_list) > 0:
                    total_load_params = torch.cat(total_load_params_list, dim=0)
                else:
                    total_load_params = torch.zeros((0,), device=fe_result.GC.device, dtype=fe_result.GC.dtype)
                num_load_params = total_load_params.numel()

                obj_part_y = torch.zeros(1, device=fe_result.GC.device, dtype=fe_result.GC.dtype)
                K_indices = self.assembly.assemble_Stiffness_Matrix(GC=fe_result.GC)[1]
                wKdp = torch.zeros_like(fe_result.GC)
                for para_idx in range(num_load_params):
                    ### compute the Jacobian sensitivity using the chain rule:
                    Ldy_now = Ldy[:, para_idx]
                    if Ldy_now.abs().sum() < 1e-8:
                        continue


                    W1 = fe_result.K_solver.solve(fe_result.K_sp, Ldy_now.cpu().numpy())
                    W1_tensor = torch.tensor(W1, dtype=fe_result.GC.dtype, device=fe_result.GC.device)
                    ### evaluate the stiffness matrix sensitivity dK/dPara using autograd
                    def get_Kdp(load_now: torch.Tensor):
                        GC_now = fe_result.GC + jacobian[:, para_idx].detach() * (load_now - load_now.detach())
                        index_now = 0
                        for load_name in load_names:
                            load = self.assembly._loads[load_name]
                            param_len = load._parameters.numel()
                            idx_now = para_idx - index_now
                            if 0 <= idx_now < param_len:
                                load._parameters[idx_now] = load_now
                                break
                            index_now += param_len
                        K_values = self.assembly.assemble_Stiffness_Matrix(GC=GC_now)[2]
                        return K_values
                    Kdp_values = torch.autograd.functional.jvp(
                        get_Kdp, total_load_params[para_idx: para_idx+1], torch.ones([1]), create_graph=False)[1]
                    Kdp = torch.sparse_coo_tensor(K_indices, Kdp_values, size=fe_result.K_sp.shape)

                    wKdp += Kdp@ W1_tensor

                    ### evaluate the stiffness matrix sensitivity dR/dPara using autograd
                    def get_Rdp(load_now: torch.Tensor):
                        GC_now = fe_result.GC + jacobian[:, para_idx].detach() * (load_now - load_now.detach())
                        index_now = 0
                        for load_name in load_names:
                            load = self.assembly._loads[load_name]
                            param_len = load._parameters.numel()
                            idx_now = para_idx - index_now
                            if 0 <= idx_now < param_len:
                                load._parameters[idx_now] = load_now
                                break
                            index_now += param_len
                        R = self.assembly.assemble_force(GC=GC_now)
                        return (R * W1_tensor).sum()

                    obj_part_y -= torch.autograd.functional.jacobian(get_Rdp, total_load_params[para_idx: para_idx+1], create_graph=True)

                if wKdp.abs().sum() > 1e-8:
                    wKdpKinv = fe_result.K_solver.solve(fe_result.K_sp, wKdp.cpu().numpy())
                else:
                    wKdpKinv = torch.zeros_like(wKdp)
                wKdpKinv_tensor = torch.tensor(wKdpKinv, dtype=fe_result.GC.dtype, device=fe_result.GC.device)
                
                obj_part_x = ((W0_tensor + wKdpKinv_tensor) * R_grad).sum()

                obj_total = obj_part_x + obj_part_y

                if idx < len(fe_results) - 1:
                    obj_total.backward(retain_graph=True)
                else:
                    obj_total.backward()

                fe_result.remove_stored_factorization()
                self._detach_recursive(fe_result)

                print(f" -Step {idx+1}/{len(fe_results)} sensitivity computed.\r", end='')

                
            sensitivity = design_vars_grad.grad.clone().detach()

        finally:            
            # Cleanup: Detach all tensors in assembly to prevent graph explosion in next run
            self._detach_recursive(self.assembly)

        print("\nAll steps sensitivity computation completed.")

        return sensitivity
        

    @classmethod
    def _detach_recursive(cls, obj: object, visited: set=None):
        """
        Recursively detach tensors to clean up the computation graph.
        For mutable containers (list, dict, objects), replaces tensors with detached versions.
        This avoids inplace detach_() errors on views.
        """
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    obj[k] = v.detach()
                else:
                    cls._detach_recursive(v, visited)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, torch.Tensor):
                    obj[i] = v.detach()
                else:
                    cls._detach_recursive(v, visited)
        elif isinstance(obj, tuple):
            for v in obj:
                cls._detach_recursive(v, visited)
        elif hasattr(obj, '__dict__'):
            # Iterate over a copy of items to avoid modification issues
            for k, v in list(obj.__dict__.items()):
                if k.startswith('__'): continue 
                if isinstance(v, torch.Tensor):
                    setattr(obj, k, v.detach())
                else:
                    cls._detach_recursive(v, visited)

    # endregion


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
            energy_new = self.get_total_energy(
                GC_now=GCnew)

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
        #         energy_new = self.get_total_energy(
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
                         GC: torch.Tensor,
                         tol_error: float):

        # record the information of the solver
        total_time = 0.0
        time_items = {'assemble': [], 'linear': [], 'line_search': [], 'step': []}

        # iteration now
        self._iter_now = 0

        # initialize the time
        t00 = time.time()

        # initialize the energy
        energy = [
            self.get_total_energy(GC_now=GC)
        ]

        dGC = torch.zeros_like(GC)

        # record the number of low alpha
        low_alpha = 0
        alpha = 0

        # begin the iteration
        while True:

            if self._iter_now > self.maximum_iteration:
                print('maximum iteration reached')
                return GC, time.time() - t00, time_items, False

            # calculate the force vector and tangential stiffness matrix
            t1 = time.time()
            R = self.assembly.assemble_force(GC=GC)
            R, K_indices, K_values = self.get_stiffness_matrix(GC_now=GC)

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
                return GC, time.time() - t00, time_items, False
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
                return GC, time.time() - t00, time_items, False

            # update the GC
            GC = GCnew

            # self.show_surface(nodes=self.nodes+RGC[0])

            # update the energy
            energynew = self.get_total_energy(
                GC_now=GC)
            energy.append(energynew)

            t4 = time.time()

            # return the index to the first line
            if self._iter_now > 1:
                print('\033[1A', end='')
                print('\033[1A', end='')
                print('\033[K', end='')

            print(  "{:^8}".format("iter") + \
                    "{:^8}".format("alpha") + \
                    "{:^8}".format("total") + \
                    "{:^15}".format("energy") + \
                    "{:^15}".format("delta_energy") + \
                    "{:^15}".format("error") + \
                    "{:^10}".format("Ktime") + \
                    "{:^10}".format("linear") + \
                    "{:^10}".format("search") + \
                    "{:^10}".format("step"))

            print(  "{:^8}".format(self._iter_now) + \
                    "{:^8.2f}".format(alpha) + \
                    "{:^8.2f}".format(t4 - t00) + \
                    "{:^15.4e}".format(energy[-1]) + \
                    "{:^15.4e}".format(energy[-1] - energy[-2]) + \
                    "{:^15.4e}".format(R.abs().max()) + \
                    "{:^10.2f}".format(t2 - t1) + \
                    "{:^10.2f}".format(t3 - t2) + \
                    "{:^10.2f}".format(t4 - t3) + \
                    "{:^10.2f}".format(t4 - t1))
            
            time_items['assemble'].append(t2 - t1)
            time_items['linear'].append(t3 - t2)
            time_items['line_search'].append(t4 - t3)
            time_items['step'].append(t4 - t1)
            
            if (dGC.abs().max() < tol_error and R.abs().max() < tol_error) or R.abs().max() < 1e-6:
                break

        
        total_time = time.time() - t00

        return GC, total_time, time_items, True

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

        if self.__low_alpha_count > 5 or R_preconditioned.abs().max() < 1e-3 or K_values_preconditioned.device.type == 'cpu':
            dx = _linear_solver.pypardiso_solver(A_indices=K_indices,
                                                 A_values=K_values_preconditioned,
                                                 b=R_preconditioned)
            self.__low_alpha_count = 0
        else:
            if self.__low_alpha_count > 0:
                dx = _linear_solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=3000)
            else:
                dx = _linear_solver.conjugate_gradient(K_indices,
                                                       K_values_preconditioned,
                                                       R_preconditioned,
                                                       x0,
                                                       tol=1e-5,
                                                       max_iter=1200)
        result = dx.to(R.dtype) / diag
        return result

    # endregion


#========= Source code for Serializable.FEAController =========#
class FEAController(Serializable):

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

    def solve(self, GC0: torch.Tensor = None, if_initialize: bool = True, *args, **kwargs):
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            bool: True if the solution converged, False otherwise.
        """
        if if_initialize:
            self.initialize()
        result = self.solver.solve(GC0=GC0, *args, **kwargs)
        return result

    def change_device(self, device: torch.device) -> None:
        """
        Recursively change the device of the finite element model and all nested objects.

        Args:
            device (torch.device): The target device.

        Returns:
            None
        """
        self._change_device_recursive(self, device)

    def _change_device_recursive(self, obj, device, visited=None):
        """
        Recursively move tensors to the target device.
        """
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    obj[k] = v.to(device)
                else:
                    self._change_device_recursive(v, device, visited)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, torch.Tensor):
                    obj[i] = v.to(device)
                else:
                    self._change_device_recursive(v, device, visited)
        elif isinstance(obj, tuple):
            for v in obj:
                self._change_device_recursive(v, device, visited)
        elif hasattr(obj, '__dict__'):
            for k, v in list(obj.__dict__.items()):
                if isinstance(v, torch.Tensor):
                    setattr(obj, k, v.to(device))
                else:
                    self._change_device_recursive(v, device, visited)

    def save_model(self, path: str, if_save_source_code: bool = False) -> None:
        """
        Save the finite element model to a file.

        Args:
            path (str): The file path to save the model.
            if_save_source_code (bool): Whether to save the source code of subclasses.

        Returns:
            None
        """
        serialized_data = self._serialize()
        if if_save_source_code: 
            np.savez_compressed(path, data = serialized_data, source_code = self._subclass_source_code)
        else:
            np.savez_compressed(path, data = serialized_data)


