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
    return torch.sum(a * b, dim=-1)


def outer(a, b):
    """calulate cross(outer) prodcut for 3-D tensor"""

    if a.size() != b.size():
        raise ValueError("input tensor shape is not same")

    out = torch.zeros_like(a)
    out[::, ::, 0] = a[::, ::, 1] * b[::, ::, 2] - a[::, ::, 2] * b[::, ::, 1]
    out[::, ::, 1] = a[::, ::, 2] * b[::, ::, 0] - a[::, ::, 0] * b[::, ::, 2]
    out[::, ::, 2] = a[::, ::, 0] * b[::, ::, 1] - a[::, ::, 1] * b[::, ::, 0]

    return out


class DihedralLayer(torch.nn.Module):
    """calulate dihedral sin and cosines
    I'm using [stackoverflow](https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python) as a reference.
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

        b0 = -1.0*(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= torch.norm(b1, dim=-1, p=2)

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - dot(b0, b1)*b1
        w = b2 - dot(b2, b1)*b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = dot(v, w)
        y = dot(outer(b1, v), w)
        rad = torch.atan2(y, x)
        print(rad)
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
        """
        size = coordinates.size()
        if len(size) == 3:
            return self._cal_dihedral(coordinates)
        elif len(size) == 4:
            return self._cal_dihedral(coordinates.view(-1, size[1], size[2])) \
                .view(size[0], size[1], size[2])
        else:
            ValueError(
                "Input tensor must 3-dim or 4-dim torch.Tensor but inputed {}"
                .format(len(size)))
