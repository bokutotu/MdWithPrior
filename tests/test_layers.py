import torch
import numpy as np

from src.layers.prior import PriorEnergyLayer
from src.layers.normalize import NormalizeLayer


def numpy_z_socore_normalize(x, sigma, myu):
    """calculate z score normalize by numpy"""
    zscore = (x-sigma)/myu
    return zscore


def test_normalize_layer():
    """test normalize layer output is collect"""
    num_atom = 10

    mean = torch.rand((num_atom))
    std = torch.rand((num_atom))

    x = torch.rand((10, 10))

    ans = numpy_z_socore_normalize(x.numpy(), mean.numpy(), std.numpy())
    output = NormalizeLayer(mean, std)(x)
    np.testing.assert_allclose(ans, output.numpy())


def test_prior_layer():
    """test Prior Layer Can Learn"""
    layer = PriorEnergyLayer(10)
    true_k = torch.randn(10)
    true_r = torch.rand(10)

    optim = torch.optim.SGD(layer.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()
    input = torch.rand((1000, 10))
    def func(x): return true_k * (x - true_r) ** 2
    first_loss = loss(layer(input), func(input))

    for i in range(1000):
        input = torch.rand(100, 10)
        out = layer(input)
        batch_loss = loss(out, func(input))
        batch_loss.backward()
        optim.step()

    input = torch.rand(10000, 10)
    output = layer(input)
    ans = func(input)
    after_train_loss = loss(output, ans)

    if after_train_loss < first_loss:
        assert True
    else:
        assert False