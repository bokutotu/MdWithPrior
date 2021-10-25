import sys

import torch


def dot(a, b):
    """calulate dot prodcut for 3D torch.Tensor

    Parameters
    ----------
    a: torch.Tensor
        a is 3 dimentional array
    b: torch.Tensor
        b is 3 dimentional array
    """
    if a.size() != b.size():
        raise ValueError("input tensor shape is not same")
    return torch.sum(a * b, dim=-1).unsqueeze(dim=-1)


class DihedralLayer(torch.nn.Module):
    """calulate dihedral sin and cosines
    I'm using
    [stackoverflow](https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python)
    as a reference.
    """

    def __init__(self, num_atoms):
        super().__init__()

        dihedral_num = num_atoms // 4 - 1

        self.register_buffer("p0_idx_phi", torch.tensor(
            list(range(2, 2 + dihedral_num * 4, 4))))
        self.register_buffer("p1_idx_phi", torch.tensor(
            list(range(4, 4 + dihedral_num * 4, 4))))
        self.register_buffer("p2_idx_phi", torch.tensor(
            list(range(5, 5 + dihedral_num * 4, 4))))
        self.register_buffer("p3_idx_phi", torch.tensor(
            list(range(6, 6 + dihedral_num * 4, 4))))

        self.register_buffer("p0_idx_psi", torch.tensor(
            list(range(4, 2 + dihedral_num * 4, 4))))
        self.register_buffer("p1_idx_psi", torch.tensor(
            list(range(5, 4 + dihedral_num * 4, 4))))
        self.register_buffer("p2_idx_psi", torch.tensor(
            list(range(6, 5 + dihedral_num * 4, 4))))
        self.register_buffer("p3_idx_psi", torch.tensor(
            list(range(8, 6 + dihedral_num * 4, 4))))

    def _cal_dihedral(self, coordinates, p0_idx, p1_idx, p2_idx, p3_idx, is_sin_cos=True):
        """calculate dihedral angles(cosine and sin) anly 3 dimentions Tensor

        Parameters
        ----------
        coordinates: torch.Tensor(3dimentional)

        Returens
        --------
        dimentions: torch.Tensor
        """
        p0 = torch.index_select(coordinates, dim=-2, index=p0_idx)
        p1 = torch.index_select(coordinates, dim=-2, index=p1_idx)
        p2 = torch.index_select(coordinates, dim=-2, index=p2_idx)
        p3 = torch.index_select(coordinates, dim=-2, index=p3_idx)

        b0 = -1.0*(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b0xb1 = torch.cross(b0, b1)
        b1xb2 = torch.cross(b2, b1)

        b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2, dim=-1)

        y = dot(b0xb1_x_b1xb2, b1) * \
            (1.0 / torch.linalg.norm(b1, ord=2, dim=-1).unsqueeze(dim=-1))
        x = dot(b0xb1, b1xb2)

        rad = torch.atan2(y, x)

        return torch.cat([torch.sin(rad), torch.cos(rad)], dim=-1), rad

    def forward(self, coordinates,):
        """calculate dihedral angles(cosine) of input coordinates

        Parameters
        ----------
        coordinates: torch.Tensor (batch, num_atoms(beads), dimentions) or
        (batch, features length, num_atoms(beads), dimentions)

        Returens
        --------
        angles: torch.Tensor
            (batch, 2 * (number of atoms - 3)) or
            (batch, length of features, 2 * (number of atoms - 3))
        """
        size = coordinates.size()
        if len(size) == 3:
            psi, phi, psi_rad, phi_rad = self._cal_psi_phi(coordinates)
            return torch.cat([psi, phi], dim=-1).view(size[0], -1), \
                    (psi_rad.view(size[0], -1), phi_rad.view(size[0], -1))
        elif len(size) == 4:
            coordinates = coordinates.view(size[0] * size[1], size[2], 3)
            psi, phi, psi_rad, phi_rad = self._cal_psi_phi(coordinates)
            return torch.cat([psi, phi], dim=-1).view(size[0], size[1], -1),\
                    (psi_rad.view(size[0], size[1], -1), phi_rad.view(size[0], size[1], -1))
        else:
            ValueError(
                "Input tensor must 3-dim or 4-dim torch.Tensor but inputed {}"
                .format(len(size)))

    def _cal_psi_phi(self, coordinates):
        psi, psi_rad = self._cal_dihedral(
            coordinates, self.p0_idx_psi, self.p1_idx_psi, self.p2_idx_psi, self.p3_idx_psi)
        phi, phi_rad = self._cal_dihedral(
            coordinates, self.p0_idx_phi, self.p1_idx_phi, self.p2_idx_phi, self.p3_idx_phi)
        return psi, phi, psi_rad, phi_rad
