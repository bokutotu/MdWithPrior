import torch


class AngleLayer(torch.nn.Module):
    """calculate angles of coordinates"""

    def _cal_angles(self, coordinates):
        """calculate angles(cosine) anly 3 dimentions Tensor

        Parameters
        ----------
        coordinates: torch.Tensor(3dimentional)

        Returens
        --------
        dimentions: torch.Tensor
        """
        basic_vec = coordinates[::, 1:-1:, ::]
        before_vec = coordinates[::, :-2:, ::]
        after_vec = coordinates[::, 2::, ::]

        inner_product = torch.sum(
            (before_vec - basic_vec) * (after_vec - basic_vec), dim=-1)

        tmp_a = torch.norm(before_vec - basic_vec, p=2, dim=-1)
        tmp_b = torch.norm(after_vec - basic_vec, p=2, dim=-1)
        norm_product = tmp_a * tmp_b

        cos_tensor = inner_product / norm_product
        return cos_tensor

    def forward(self, coordinates):
        """calculate bond angles(cosine) of input coordinates 

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
            return self._cal_angles(coordinates)

        elif len(size) == 4:
            return self._cal_angles(
                coordinates.view(size[0] * size[1], size[2], size[3]))\
                    .view(size[0], size[1], size[2]-2)
        else:
            ValueError(
                "Input tensor must 3-dim or 4-dim torch.Tensor but inputed {}" \
                    .format(len(size)))
