from src.statics import get_statics
from src.features.angles import AngleLayer
from src.features.dihedral import DihedralLayer
from src.features.length import LengthLayer

import numpy as np
import torch


def test_statics():
    """test if statics is work fine,
    This test assumes that AngleLayer, DihedralLayer, and LengthLayer 
    work correctly"""
    coordinates = np.load("./tests/data/ala2_coordinates.npy")
    coordinates = torch.from_numpy(coordinates)
    statics = get_statics(coordinates)

    # dim is (lentgh of sim, number of atom(beads), 3)
    angle = AngleLayer()(coordinates).numpy() 
    dihedral = DihedralLayer()(coordinates).numpy()
    length = LengthLayer()(coordinates).numpy()

    angle_mean = np.mean(angle, axis=0)
    angle_std = np.std(angle, axis=0)
    dihedral_mean = np.mean(dihedral, axis=0)
    dihedral_std = np.std(dihedral, axis=0)
    length_mean = np.mean(length, axis=0)
    length_std = np.std(length, axis=0)

    np.testing.assert_allclose(statics["angle"]["mean"].numpy(), angle_mean, rtol=1e-4)
    np.testing.assert_allclose(statics["angle"]["std"].numpy(), angle_std, rtol=1e-4)
    np.testing.assert_allclose(statics["length"]["mean"].numpy(), length_mean, rtol=1e-4)
    np.testing.assert_allclose(statics["length"]["std"].numpy(), length_std, rtol=1e-4)
    np.testing.assert_allclose(statics["dihedral"]["mean"].numpy(), dihedral_mean, rtol=1e-4)
    np.testing.assert_allclose(statics["dihedral"]["std"].numpy(), dihedral_std, rtol=1e-4)
