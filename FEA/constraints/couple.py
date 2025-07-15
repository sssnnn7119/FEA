import numpy as np
import torch
from .base import BaseConstraint

class Couple(BaseConstraint):

    def __init__(self, indexNodes: np.ndarray, rp_name: str) -> None:
        super().__init__()
        self.indexNodes = np.sort(indexNodes.reshape([-1]))
        self.rp_name = rp_name
        self.rp_index: int

    def initialize(self, fea):
        super().initialize(fea)
        self.rp_index = self._fea.reference_points[self.rp_name]._RGC_index
        self.ref_location = self._fea.nodes[
            self.indexNodes] - self._fea.reference_points[self.rp_name].node

    def modify_RGC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply the couple constraint to the displacement vector
        """
        RGC[0][self.indexNodes] = RGC[self.rp_index][:3] + self._rotation3d(
            RGC[self.rp_index][3:], self.ref_location) - self.ref_location
        RGC[1][self.indexNodes] = RGC[self.rp_index][3:]

        return RGC

    def modify_RGC_linear(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        RGC[0][self.indexNodes] = RGC[
            self.rp_index][:3] + self._rotation3d_linear(
                RGC[self.rp_index][3:], self.ref_location) - self.ref_location
        RGC[1][self.indexNodes] = RGC[self.rp_index][3:]

        return RGC

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[0][self.indexNodes] = False
        RGC_remain_index[self.rp_index][:] = True
        return RGC_remain_index

    def modify_R_K(self, RGC: list[torch.Tensor], R0: torch.Tensor,
                   K_indices: torch.Tensor, K_values: torch.Tensor):
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

        R0 = R0[:self._fea.RGC_list_indexStart[1]].view(-1, 3)

        # basic derivatives
        y = self.ref_location
        v = RGC[self.rp_index][:3]
        z = RGC[self.rp_index][3:]
        theta = z.norm()
        w = (z / z.norm())

        epsilon_indices = [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1],
                           [2, 1, 2, 0, 1, 0]]
        epsilon_values = [1, -1, -1, 1, 1, -1]

        # region

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

        #endregion

        # R
        # region
        Rrest = R0[self.indexNodes]

        Edotv = Rrest.sum(dim=0)
        Edotz = torch.einsum('bj,bjp->p', Rrest, Ydot)

        R = torch.sparse_coo_tensor(indices=torch.arange(
            self._fea.RGC_list_indexStart[self.rp_index],
            self._fea.RGC_list_indexStart[self.rp_index] + 6).unsqueeze(0),
                                    values=torch.cat([Edotv, Edotz], dim=0),
                                    size=[self._fea.RGC_list_indexStart[-1]])
        # endregion

        # K
        # region
        ## first, get the K of the rest part in index1

        index = torch.where(
            torch.isin((K_indices[1] // 3),
                       torch.tensor(self.indexNodes.tolist())))

        sort_index = torch.argsort(K_indices[1][index] // 3)

        index = index[0][sort_index]
        indice1 = K_indices[0][index] // 3
        indice2 = K_indices[0][index] % 3
        indice30 = K_indices[1][index] // 3
        indice3 = torch.unique_consecutive(indice30, return_inverse=True)[1]
        indice4 = K_indices[1][index] % 3

        Rdotv_indices = torch.stack([indice1, indice2, indice4], dim=0)
        Rdotv_indices_flatten = Rdotv_indices[0] * 9 + Rdotv_indices[
            1] * 3 + Rdotv_indices[2]
        Rdotv_values = K_values[index]
        Rdotv = torch.zeros([RGC[0].shape[0] * 3 * 3]).scatter_add_(
            0, Rdotv_indices_flatten,
            Rdotv_values).reshape(RGC[0].shape[0], 3, 3)

        Rdotz_indices = torch.stack([
            indice1.reshape(-1, 1).repeat(1, 3).flatten(),
            indice2.reshape(-1, 1).repeat(1, 3).flatten(),
            torch.tensor([0, 1, 2]).reshape([1, 3]).repeat(
                indice1.shape[0], 1).flatten()
        ],
                                    dim=0)
        Rdotz_indices_flatten = Rdotz_indices[0] * 9 + Rdotz_indices[
            1] * 3 + Rdotz_indices[2]
        Rdotz_values = (K_values[index].unsqueeze(-1) *
                        Ydot.view(-1, 3)[indice4 + indice3 * 3]).flatten()
        Rdotz = torch.zeros([RGC[0].shape[0] * 3 * 3]).scatter_add(
            0, Rdotz_indices_flatten,
            Rdotz_values).reshape(RGC[0].shape[0], 3, 3)

        Edotvv = Rdotv[self.indexNodes].sum(dim=0)
        Edotzv = torch.einsum('biq,bip->pq', Rdotv[self.indexNodes], Ydot)

        Edotzz = torch.einsum('biq,bip->pq', Rdotz[self.indexNodes],
                              Ydot) + torch.einsum('ai,aipq->pq', Rrest, Ydot2)
        # combine the indices and values
        indices = []
        values = []

        ## for Rv
        indice_Rv = Rdotv_indices
        index1 = indice_Rv[0] * 3 + indice_Rv[1]
        index2 = self._fea.RGC_list_indexStart[self.rp_index] + indice_Rv[2]
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(Rdotv_values)
        indices.append(torch.stack([index2, index1], dim=0))
        values.append(Rdotv_values)
        ## for Rz
        indice_Rz = Rdotz_indices
        index1 = indice_Rz[0] * 3 + indice_Rz[1]
        index2 = indice_Rz[2] + self._fea.RGC_list_indexStart[self.rp_index] + 3
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
        index1 = indice_Edot2[0] + self._fea.RGC_list_indexStart[self.rp_index]
        index2 = indice_Edot2[1] + self._fea.RGC_list_indexStart[self.rp_index]
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(mat66.flatten())

        # combine the indices and values
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values, dim=0)
        #endregion

        return R, indices, values

    def modify_R(self, RGC: list[torch.Tensor],
                 R0: torch.Tensor) -> torch.Tensor:
        """
        Modify the R

        Args:
            indexStart (list[int]): The starting indices for each node.
            U (list[torch.Tensor]): The displacement vector for each node.
            R (torch.Tensor): The global force vector.
            K (torch.Tensor): The global stiffness matrix.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The modified R and K tensors.
        """

        R0 = R0[:self._fea.RGC_list_indexStart[1]].view(-1, 3)

        # basic derivatives
        y = self.ref_location
        z = RGC[self.rp_index][3:]
        theta = z.norm()
        w = (z / z.norm())

        epsilon_indices = [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1],
                           [2, 1, 2, 0, 1, 0]]
        epsilon_values = [1, -1, -1, 1, 1, -1]

        # region

        der_theta = -y * torch.sin(theta) + w.view(
            1, 3) * (w.view(1, 3) * y).sum(dim=1).reshape(-1, 1) * torch.sin(
                theta) + torch.cross(w.view(1, 3), y, dim=1) * torch.cos(theta)

        der_w = (torch.einsum('al,i->ail', y, w)) * (1 - torch.cos(theta))
        temp = (1 - torch.cos(theta)) * (w.view(1, 3) * y).sum(dim=1).flatten()
        for i in range(3):
            der_w[:, i, i] += temp
        for i in range(6):
            der_w[:, epsilon_indices[0][i],
                  epsilon_indices[2][i]] -= epsilon_values[i] * torch.sin(
                      theta) * y[:, epsilon_indices[1][i]]

        wdot = -torch.einsum('i,p->ip', z, z) / theta**3 + torch.eye(3) / theta
        thetadot = w

        Ydot = torch.einsum('bjl,lp->bjp', der_w, wdot) + torch.einsum(
            'bj,p->bjp', der_theta, thetadot)

        Rrest = R0[self.indexNodes]

        Edotv = Rrest.sum(dim=0)
        Edotz = torch.einsum('bj,bjp->p', Rrest, Ydot)

        R = torch.sparse_coo_tensor(indices=[
            np.arange(self._fea.RGC_list_indexStart[self.rp_index],
                      self._fea.RGC_list_indexStart[self.rp_index] + 6)
        ],
                                    values=torch.cat([Edotv, Edotz], dim=0),
                                    size=[self._fea.RGC_list_indexStart[-1]])
        return R

    def modify_K(
        self,
        RGC: list[torch.Tensor],
        R0: torch.Tensor,
        K0: torch.Tensor,
    ):
        """
        Modify the K

        Args:
            indexStart (list[int]): The starting indices for each node.
            U (list[torch.Tensor]): The displacement vector for each node.
            R (torch.Tensor): The global force vector.
            K (torch.Tensor): The global stiffness matrix.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The modified R and K tensors.
        """

        # basic derivatives
        y = self.ref_location
        v = RGC[self.rp_index][:3]
        z = RGC[self.rp_index][3:]
        theta = z.norm()
        w = (z / z.norm())

        epsilon_indices = [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1],
                           [2, 1, 2, 0, 1, 0]]
        epsilon_values = [1, -1, -1, 1, 1, -1]

        # region

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

        #endregion

        # K
        # region
        ## first, get the K of the rest part in index1

        index = torch.where(
            torch.isin((K0.indices()[1] // 3),
                       torch.tensor(self.indexNodes.tolist())))
        sort_index = torch.argsort(K0.indices()[1][index] // 3)
        index = index[0][sort_index]
        indice1 = K0.indices()[0][index] // 3
        indice2 = K0.indices()[0][index] % 3
        indice30 = K0.indices()[1][index] // 3
        indice3 = torch.cat([
            torch.ones((indice30 == self.indexNodes[i]).sum()) * i
            for i in range(len(self.indexNodes))
        ])

        indice4 = K0.indices()[1][index] % 3

        K_rest = torch.sparse_coo_tensor(
            indices=torch.stack([indice1, indice2, indice3, indice4], dim=0),
            values=K0.values()[index],
            size=[RGC[0].shape[0], 3,
                  len(self.indexNodes), 3]).coalesce()

        Rdotv = K_rest.sum(dim=2).to_dense()
        Rdotz = (_sparse_reshape(
            K_rest, [RGC[0].shape[0] * 3,
                     len(self.indexNodes) * 3]) @ Ydot.view(-1, 3)).view(
                         RGC[0].shape[0], 3, 3)
        Edotvv = Rdotv[self.indexNodes].sum(dim=0)
        Edotzv = torch.einsum('biq,bip->pq', Rdotv[self.indexNodes], Ydot)

        Edotzz = torch.einsum('biq,bip->pq',
                              Rdotz[self.indexNodes], Ydot) - torch.einsum(
                                  'ai,aipq->pq', R0[self.indexNodes], Ydot2)
        # combine the indices and values
        indices = []
        values = []
        Rdotv = Rdotv.to_sparse_coo()
        Rdotz = Rdotz.to_sparse_coo()

        ## for Rv
        indice_Rv = Rdotv.indices()
        index1 = indice_Rv[0] * 3 + indice_Rv[1]
        index2 = self._fea.RGC_list_indexStart[self.rp_index] + indice_Rv[2]
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(Rdotv.values())
        indices.append(torch.stack([index2, index1], dim=0))
        values.append(Rdotv.values())
        ## for Rz
        indice_Rz = Rdotz.indices()
        index1 = indice_Rz[0] * 3 + indice_Rz[1]
        index2 = indice_Rz[2] + self._fea.RGC_list_indexStart[self.rp_index] + 3
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(Rdotz.values())
        indices.append(torch.stack([index2, index1], dim=0))
        values.append(Rdotz.values())
        ## for Edot2
        mat66 = torch.zeros([6, 6])
        mat66[:3, :3] = Edotvv
        mat66[3:, 3:] = Edotzz
        mat66[3:, :3] = Edotzv
        mat66[:3, 3:] = Edotzv.transpose(0, 1)
        mat66 = mat66.to_sparse_coo()
        indice_Edot2 = mat66.indices()
        index1 = indice_Edot2[0] + self._fea.RGC_list_indexStart[self.rp_index]
        index2 = indice_Edot2[1] + self._fea.RGC_list_indexStart[self.rp_index]
        indices.append(torch.stack([index1, index2], dim=0))
        values.append(mat66.values())

        # combine the indices and values
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values, dim=0)
        #endregion

        return indices, values

    def _rotation3d(self, rotation_vector: torch.Tensor,
                    vector0: torch.Tensor):
        """
        Rotate a 3D vector by a rotation vector
        :param rotation_vector: rotation vector (3,)
        :param vector0: 3D vector (n, 3)
        :return: 3D vector (n, 3)
        """
        vector0 = vector0.view(-1, 3)
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

    def _rotation3d_linear(self, rotation_vector: torch.Tensor,
                           vector0: torch.Tensor):
        """
        Rotate a 3D vector by a rotation vector
        :param rotation_vector: rotation vector (3,)
        :param vector0: 3D vector (n, 3)
        :return: 3D vector (n, 3)
        """
        vector0 = vector0.view(-1, 3)
        theta = torch.norm(rotation_vector)
        if theta == 0:
            return vector0
        else:
            rotation_vector = rotation_vector / theta
            rotation_vector = rotation_vector.view(1, 3)
            vector1 = vector0 + torch.cross(rotation_vector, vector0,
                                            dim=1) * theta
        return vector1


def _sparse_reshape(sparse_tensor, new_shape):

    # coalesce the COO sparse tensor
    sparse_tensor = sparse_tensor.coalesce()

    # get the indices and values of the COO sparse tensor
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # get the number of non-zero elements
    nnz = values.numel()

    # get the original shape and new shape of the COO sparse tensor
    original_shape = sparse_tensor.shape

    # get the true index of the COO sparse tensor
    true_indices = torch.zeros([nnz], dtype=torch.long)
    for i in range(len(original_shape)):
        true_indices = true_indices * original_shape[i] + indices[i]

    # calculate the new indices of the COO sparse tensor
    new_indices = torch.zeros([len(new_shape), nnz], dtype=torch.long)
    for i in range(len(new_shape) - 1, -1, -1):
        new_indices[i, :] = true_indices % new_shape[i]
        true_indices = true_indices // new_shape[i]

    # construct the new COO sparse tensor
    new_sparse_tensor = torch.sparse_coo_tensor(new_indices,
                                                values,
                                                size=new_shape)

    return new_sparse_tensor
