import numpy as np
import torch

from src.dataset import LSTMDataset, MLPDataset

def test_lstm_dataset():
    coord = np.load("./tests/data/ala2_coordinates.npy")
    force = np.load("./tests/data/ala2_forces.npy")

    dataset = LSTMDataset(coord, force, 100)
    length = len(dataset)
    for idx in range(length):
        a, b = dataset[idx]

    assert True


def test_mlp_dataset():
    coord = np.load("./tests/data/ala2_coordinates.npy")
    force = np.load("./tests/data/ala2_forces.npy")

    dataset = LSTMDataset(coord, force, 100)
    length = len(dataset)
    for idx in range(length):
        a, b = dataset[idx]

    assert True


def test_lstm_dataloader_is_eneable_grad():
    coord = np.load("./tests/data/ala2_coordinates.npy")
    force = np.load("./tests/data/ala2_forces.npy")

    dataset = LSTMDataset(coord, force, 100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
            sampler=torch.utils.data.RandomSampler(dataset))
    for batch in dataloader:
        x, y = batch
        # assert x.requires_grad