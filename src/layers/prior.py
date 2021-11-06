import torch


class PriorEnergyLayer(torch.nn.Module):
    """Prior Energy Layer in Machine Learning of Coarse-Grained Molecular 
    Dynamics Force Fields

    Parameters
    ----------
    k: torch.tensor shape(number of atoms(beads))
        In Paper use this coeffiecint as K

    r: torch.tensor shape (number of atoms (beads))
        In Paper use this coeffiecint as r or θ

    """

    def __init__(self, size, r=None):
        super().__init__()
        if r is not None:
            print("set mean parameters")
        else:
            print("not set mean")
        self.k = torch.nn.Parameter(torch.randn(size), requires_grad=True)
        if r is None:
            self.r = torch.nn.Parameter(torch.randn(size), requires_grad=True)
        else:
            self.r = torch.nn.Parameter(r)

    def forward(self, x):
        return torch.sum((self.k * (x - self.r))**2, dim=-1).unsqueeze(dim=-1)
