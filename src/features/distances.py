import torch


class DistanceLayer(torch.nn.Module):
    """Calulate distances of coordinates


    """

    def _call_distances(self, coordinates):
        """calculate distances anly 3 dimentions Tensor

        Parameters
        ----------
        coordinates: torch.Tensor(3dimentional)

        Returens
        --------
        dimentions: torch.Tensor
        """
        return torch.sqrt(
            torch.sum(
                torch.pow(
                    coordinates[::, 0:-1:, ::] - coordinates[::, 1::, ::], 2),
                dim=-1),
        )

    def forward(self, coordinates):
        """calculate bond length of input coordinates 

        Parameters
        ----------
        coordinates: torch.Tensor (batch, num_atoms(beads), dimentions) or 
        (batch, features length, num_atoms(beads), dimentions)

        Returens
        --------
        distances: torch.Tensor
        """

        size = coordinates.size()
        if len(size) == 3:
            return self._call_distances(coordinates)

        elif len(size) == 4:
            return self._call_distances(coordinates.view(-1, size[1], size[2])).view(size[0], size[1], size[2])
        else:
            ValueError(
                "Input tensor must 3-dim or 4-dim torch.Tensor but inputed {}".format(len(size)))
