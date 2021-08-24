import torch
from torch.utils.data import sampler

from omegaconf import OmegaConf
import numpy as np

from src.model import CGnet
from src.dataset import LSTMDataset


def test_epoch():
    coord = np.load("./tests/data/ala2_coordinates.npy")
    force = np.load("./tests/data/ala2_forces.npy")

    dataset = LSTMDataset(coord, force, 64)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        sampler=sampler.RandomSampler(dataset)
    )

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
    net = CGnet(conf, 5, torch.rand(3), torch.rand(3),
                torch.rand(2 * (5 - 3)), torch.rand(2 * (5 - 3)),
                torch.rand(4), torch.rand(4))

    optim = torch.optim.SGD(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    batch_counter = 0

    for (idx, batch) in enumerate(dataloader):
        if batch_counter == 5:
            break
        optim.zero_grad()
        x, y = batch
        print(x.size())
        out, _ = net(x)
        loss = loss_func(out, y)
        loss.backward()
        optim.step()
        batch_counter += 1
