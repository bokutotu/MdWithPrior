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

    def _cal_dihedral(self, coordinates):
        """calculate dihedral angles(cosine and sin) anly 3 dimentions Tensor

        Parameters
        ----------
        coordinates: torch.Tensor(3dimentional)

        Returens
        --------
        dimentions: torch.Tensor
        """
        p0 = coordinates[::, 0:-3:, ::]
        p1 = coordinates[::, 1:-2:, ::]
        p2 = coordinates[::, 2:-1:, ::]
        p3 = coordinates[::, 3::, ::]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 = b1 / (torch.linalg.norm(b1, ) + sys.float_info.epsilon)

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - dot(b0, b1) * b1
        w = b2 - dot(b2, b1) * b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = dot(v, w)
        y = dot(torch.cross(b1, v, dim=-1), w)
        rad = torch.atan2(y, x)
        return torch.cat([torch.sin(rad), torch.cos(rad)], dim=-1)

    def forward(self, coordinates):
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
            return self._cal_dihedral(coordinates).view(size[0], 2 * (size[1] - 3))
        elif len(size) == 4:
            return self._cal_dihedral(
                coordinates.view(size[0] * size[1], size[2], 3)) \
                .view(size[0], size[1], 2 * (size[2] - 3))
        else:
            ValueError(
                "Input tensor must 3-dim or 4-dim torch.Tensor but inputed {}"
                .format(len(size)))
