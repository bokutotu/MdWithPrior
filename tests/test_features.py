import torch
import numpy as np

from src.features.length import LengthLayer
from src.features.angles import AngleLayer
from src.features.dihedral import DihedralLayer


def test_cal_distance():
    """test calculation result is collect"""
    input = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [3., 2., 1.]], ).view(1, 3, 3)
    layer = LengthLayer()
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
    input = torch.tensor(array, requires_grad=True).view(1, 4, 3)
    layer = DihedralLayer()
    output = layer(input)

    rad = np.radians(-71.21515)

    ans = np.array([[np.sin(rad), np.cos(rad)]], dtype=np.float32)
    np.testing.assert_allclose(output.detach().numpy(), ans, rtol=1e-6)


def test_can_cal_grad_features():
    angle_layer = AngleLayer()
    length_layer = LengthLayer()
    dihedral_layer = DihedralLayer()

    input_tesnor = torch.rand((10, 10, 3), requires_grad=True)

    angle = angle_layer(input_tesnor)
    length = length_layer(input_tesnor)
    dihedral = dihedral_layer(input_tesnor)

    features = torch.cat([angle, length], dim=-1)

    grad = torch.autograd.grad(torch.sum(angle), input_tesnor, 
                create_graph=True, retain_graph=True)

    grad = torch.autograd.grad(torch.sum(length), input_tesnor, 
                create_graph=True, retain_graph=True)

    grad = torch.autograd.grad(torch.sum(dihedral), input_tesnor, 
                create_graph=True, retain_graph=True)

    grad = torch.autograd.grad(torch.sum(features), input_tesnor, 
                create_graph=True, retain_graph=True)