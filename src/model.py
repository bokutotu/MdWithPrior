import torch
import torch.nn as nn

from hydra.utils import instantiate

from src.layers.prior import PriorEnergyLayer
from src.layers.normalize import NormalizeLayer

from src.features.angles import AngleLayer
from src.features.dihedral import DihedralLayer
from src.features.length import LengthLayer


class CGnet(nn.Module):
    """A class for learning and reasoning about CGnet.

    As described in the paper, the neural network that calculates the energy of
    sparse viewing uses the bond length, bond angle, and dihedral angle
    calculated from the coordinates as input. In addition, Prior Energy can be
    used selectively for the bond length, bond angle, and dihedral angle,
    respectively.

    It is also possible, depending on the options, to adapt the previously
    obtained standardization with zscore for all inputs.

    In this model, time series is supported.

    Parameters
    ----------
    config: omegaconf
        config got by hydra.
    num_atom: int
        number of atom (or beads) use to train
    angle_mean: torch.tensor
        bond angle mean
    angle_std: torch.tensor
        bond angle std
    length_mean: torch.tensor
        bond length mean
    length_std: torch.tensor
        bond length std
    dihedral_mean: torch.tensor
        dihedral mean
    dihedral_std: torch.tensor
        dihedral std
    """

    def __init__(
            self, config, num_atom,
            angle_mean=None, angle_std=None,
            dihedral_mean=None, dihedral_std=None,
            length_mean=None, length_std=None, cmap=None, 
            forces_std=None, forces_mean=None):
        super().__init__()
        self.config = config

        self.cal_angle_layer = AngleLayer()
        self.cal_dihedral_layer = DihedralLayer(num_atom)
        self.cal_length_layer = LengthLayer()

        # define Prior Energy Layer
        if config.is_angle_prior:
            self.angle_prior_layer = PriorEnergyLayer(num_atom - 2,
            angle_mean if config.use_prior_mean else None)
        if config.is_length_prior:
            self.length_prior_layer = PriorEnergyLayer(
                num_atom - 1, length_mean if config.use_prior_mean else None)
        if config.is_dihedral_prior:
            self.dihedral_prior_layer = PriorEnergyLayer(
                dihedral_mean.size()[0],
                dihedral_mean if config.use_prior_mean else None)
        if config.is_cmap:
            self.cmap = cmap

        if config.is_normalize:
            self.angle_normalize_layer = self._define_normalize_layer(
                angle_mean, angle_std
            )
            self.length_normalize_layer = self._define_normalize_layer(
                length_mean, length_std
            )
            self.dihedral_normalize_layer = self._define_normalize_layer(
                dihedral_mean, dihedral_std
            )
        self.net = instantiate(config.models,
                               input_size=self._get_input_dim_from_num_atom(num_atom))
        self.is_use_cudnn = not config.models._target_.split(
            ".")[-1] in ["LSTM", "GRU"]
        if not self.is_use_cudnn:
            print("CUDNN disabled")

        self.register_buffer("std", torch.tensor(forces_std))
        self.register_buffer("mean", torch.tensor(forces_mean))

    def _get_input_dim_from_num_atom(self, num_atom):
        input_dim = 0
        input_dim += num_atom - 1  # length dim
        input_dim += num_atom - 2  # angle dim
        input_dim += ((num_atom // 4) - 1) * 4  # dihedral number
        return input_dim

    def _define_normalize_layer(self, mean, std):
        if mean is None:
            raise ValueError(
                "in config is_normalize True You should set mean")
        if std is None:
            raise ValueError(
                "in config is_normalize True You should set std")
        if std.size() != mean.size():
            raise ValueError("std and mean shape is not same",
                             "std size: {} mean size: {}"
                             .format(std.size(), mean.size()))

        return NormalizeLayer(mean, std)

    def forward(self, x, is_use_NN=True):
        # x shape is (batch size, number of atom(beads), 3) or
        # (batch size, length of featuers, number of atom(beads), 3)

        # cal features
        angle = self.cal_angle_layer(x)
        length = self.cal_length_layer(x)
        dihedral, dihedral_rad = self.cal_dihedral_layer(x)

        if self.config.is_normalize:
            angle = self.angle_normalize_layer(angle)
            length = self.length_normalize_layer(length)
            dihedral = self.dihedral_normalize_layer(dihedral)

        if is_use_NN:
            net_input = torch.cat([angle, length, dihedral], dim=-1)
            with torch.backends.cudnn.flags(self.is_use_cudnn):
                energy = self.net(net_input)

        # output shape is
        # (batch size, 1) or
        # (batch size, length of features, 1)

        if self.config.is_angle_prior:
            if is_use_NN:
                energy = energy + self.angle_prior_layer(angle)
            else:
                energy = self.angle_prior_layer(angle)
        if self.config.is_length_prior:
            energy = energy + self.length_prior_layer(length)
        if self.config.is_dihedral_prior:
            energy = energy + self.dihedral_prior_layer(dihedral)
        if self.config.is_cmap:
            energy = energy + self.cmap(*dihedral_rad)

        force = torch.autograd.grad(-torch.sum(energy),
                                    x,
                                    create_graph=True,
                                    retain_graph=True)

        force = force[0]
        force = self.std * force + self.mean
        return force, energy
