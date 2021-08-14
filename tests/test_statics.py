from src.statics import get_statics

import numpy as np
import torch


def test_statics():
    coordinates = np.load("./tests/data/ala2_coordinates.npy")
    statics = get_statics(coordinates)

    length_mean = np.array([1.3293, 1.4654, 1.5436, 1.3354])
    length_std = np.array([0.0253, 0.0300, 0.0305, 0.0251])

    angle_mean = np.array([2.1725, 1.9631, 2.0426])
    angle_std = np.array([0.0551, 0.0576, 0.0509])

    np.testing.assert_allclose(statics["angle"]["mean"].numpy(), angle_mean)
    np.testing.assert_allclose(statics["angle"]["std"].numpy(), angle_std)
    np.testing.assert_allclose(statics["length"]["mean"].numpy(), length_mean)
    np.testing.assert_allclose(statics["length"]["std"].numpy(), length_std)
