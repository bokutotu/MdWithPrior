import torch
from omegaconf import OmegaConf

from src.layers.lstm import LSTM
from src.layers.mlp import MLP
from src.layers.prior import PriorEnergyLayer

from src.features.angles import AngleLayer
from src.features.length import LengthLayer
from src.features.dihedral import DihedralLayer

from src.model import CGnet


def test_lstm_is_grad():
    net = LSTM(
        num_layers=2, dropout=0.0, input_size=100,
        hidden_size=100, output_size=1
    )

    fake_input = torch.rand((10, 64, 100), requires_grad=True)
    output = net(fake_input)

    grad_res = torch.autograd.grad(-1 * torch.sum(
        output), fake_input, create_graph=True, retain_graph=True)
    
    ans = torch.rand((10, 64, 100), requires_grad=True)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    optim.zero_grad()

    loss = torch.nn.MSELoss()(ans, fake_input)
    loss.backward()
    optim.step()


def test_mlp_is_grad():
    net = MLP(
        num_layers=2, dropout=0.0, input_size=100,
        hidden_size=100, output_size=1, activate_function="ReLU"
    )

    fake_input = torch.rand((64, 100), requires_grad=True)
    output = net(fake_input)

    grad_res = torch.autograd.grad(-1 * torch.sum(
        output), fake_input, create_graph=True, retain_graph=True)
    
    ans = torch.rand((64, 100), requires_grad=True)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    optim.zero_grad()

    loss = torch.nn.MSELoss()(ans, fake_input)
    loss.backward()
    optim.step()


def test_prior_is_grad():
    net = PriorEnergyLayer(100)

    fake_input = torch.rand((64, 100), requires_grad=True)
    output = net(fake_input)

    grad_res = torch.autograd.grad(-1 * torch.sum(
        output), fake_input, create_graph=True, retain_graph=True)
    
    ans = torch.rand((64, 100), requires_grad=True)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    optim.zero_grad()

    loss = torch.nn.MSELoss()(ans, fake_input)
    loss.backward()
    optim.step()


def test_angle_is_grad():
    net = AngleLayer()
    fake_input = torch.rand((64, 100, 3), requires_grad=True)
    output = net(fake_input)

    grad_res = torch.autograd.grad(-1 * torch.sum(
        output), fake_input, create_graph=True, retain_graph=True)
    
    ans = torch.rand((64, 100, 3), requires_grad=True)

    loss = torch.nn.MSELoss()(ans, fake_input)
    loss.backward()


def test_angle_is_grad():
    net = LengthLayer()
    fake_input = torch.rand((64, 100, 3), requires_grad=True)
    output = net(fake_input)

    grad_res = torch.autograd.grad(-1 * torch.sum(
        output), fake_input, create_graph=True, retain_graph=True)
    
    ans = torch.rand((64, 100, 3), requires_grad=True)

    loss = torch.nn.MSELoss()(ans, fake_input)
    loss.backward()


def test_angle_is_grad():
    net = DihedralLayer()
    fake_input = torch.rand((64, 100, 3), requires_grad=True)
    output = net(fake_input)

    grad_res = torch.autograd.grad(-1 * torch.sum(
        output), fake_input, create_graph=True, retain_graph=True)
    
    ans = torch.rand((64, 100, 3), requires_grad=True)

    loss = torch.nn.MSELoss()(ans, fake_input)
    loss.backward()


def test_combine_grad():
    conf = {
        "is_angle_prior": True,
        "is_length_prior": True,
        "is_dihedral_prior": True,
        "is_normalize": True,
        "models": {
            "_target_": "src.layers.lstm.LSTM",
            "num_layers": 5,
            "dropout": 0.0, 
            "hidden_size": 256,
            "output_size": 1
        }
    }

    conf = OmegaConf.create(conf)
    net = CGnet(conf, 100, torch.rand(98), torch.rand(98), 
        torch.rand(2 * (100 - 3)), torch.rand(2 * (100 - 3)), 
        torch.rand(99), torch.rand(99))

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    optimizer.zero_grad()

    input_tensor = torch.rand((16, 64, 100, 3), requires_grad=True)
    output, _ = net(input_tensor)
    loss = loss_func(output, input_tensor)
    loss.backward()
    optimizer.step()