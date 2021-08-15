import torch
from src.features.angles import AngleLayer
from src.features.length import LengthLayer
from src.features.dihedral import DihedralLayer


def get_statics(coordinates):
    """get Statics about mean, std about length, bond angles and dihedral angles

    coordinates: numpy.array
    """
    angles = AngleLayer()(coordinates)
    lengths = LengthLayer()(coordinates)
    dihedrals = DihedralLayer()(coordinates)

    angle_mean = torch.mean(angles, dim=0)
    length_mean = torch.mean(lengths, dim=0)
    dihedral_mean = torch.mean(dihedrals, dim=0)

    angle_std = torch.std(angles, dim=0)
    length_std = torch.std(lengths, dim=0)
    dihedral_std = torch.std(dihedrals, dim=0)

    return {"angle": {"mean": angle_mean, "std": angle_std},
            "length": {"mean": length_mean, "std": length_std},
            "dihedral": {"mean": dihedral_mean, "std": dihedral_std}}
