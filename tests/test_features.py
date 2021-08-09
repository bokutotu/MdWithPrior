import torch
import numpy as np

from src.features.distances import DistanceLayer
from src.features.angles import AngleLayer
from src.features.dihedral import DihdrralLayer


def test_cal_distance():
    """test calculation result is collect"""
    input = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [3., 2., 1.]], ).view(1, 3, 3)
    layer = DistanceLayer()
    output = layer(input)
    ans = torch.tensor([[1.0, 3.0]])
    np.testing.assert_allclose(ans.numpy(), output.numpy())


def test_cal_angles():
    """test calculation result is collect"""
    input = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [3., 2., 1.]], ).view(1, 3, 3)
    layer = AngleLayer()
    output = layer(input)
    ans = torch.tensor([[-2 / 3]])
    np.testing.assert_allclose(ans.numpy(), output.numpy())


def test_cal_dihedral():
    """test calculation result is collect"""
    array = [
            [24.969, 13.428, 30.692],
            [24.044, 12.661, 29.808],
            [22.785, 13.482, 29.543],
            [21.951, 13.670, 30.431],
    ]
    input = torch.tensor(array).view(1, 4, 3)
    layer = DihdrralLayer()
    output = layer(input)
    ans = torch.tensor([[np.sin(-71.21515), np.cos(-71.21515)]])
    np.testing.assert_allclose(ans.numpy(), ans.numpy())
