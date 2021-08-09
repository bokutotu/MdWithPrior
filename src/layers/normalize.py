import torch


class NormalizeLayer(torch.nn.Module):
    """Normalize Input Tensor using mean, std

    Parameters
    ----------
    mean: torch.Tensor
        mean
    std: torch.Tensor
        std
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        """normalizate

        do (x - mean) / std

        Parameters
        ----------
        x: torch.tensor
            2-D or 3-D tensor 
            if 2-D -> (batch, features)
            if 3-D -> (batch, length of features, features)
        """

        x = (x - self.mean) / self.std
        return x