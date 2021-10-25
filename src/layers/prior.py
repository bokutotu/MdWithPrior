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

    def __init__(self, size):
        super().__init__()
        self.k = torch.nn.Parameter(torch.randn(size), requires_grad=True)
        self.r = torch.nn.Parameter(torch.randn(size), requires_grad=True)

    def forward(self, x):
        return torch.sum((self.k * (x - self.r))**2, dim=-1).unsqueeze(dim=-1)
