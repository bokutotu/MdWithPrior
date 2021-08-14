from typing import List, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MLPDataset(Dataset):

    def __init__(self, coordinates, forces):
        self.coordinates = coordinates
        self.forces = forces

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.coordinates[idx], requireds_grad=True),
            torch.tensor(self.forces[idx], requireds_grad=True)
        )


class LSTMDataset(Dataset):

    def __init__(self, coordinates, forces, features_length):
        self.coordinates = coordinates
        self.forces = forces
        self.features_length = features_length

    def __len__(self):
        return len(self.coordinates) - self.features_length

    def __getitem__(self, idx):
        return (
            torch.tensor(
                self.coordinates[idx:idx+self.features_length], requireds_grad=True),
            torch.tensor(
                self.forces[idx:idx+self.features_length], requireds_grad=True)
        )
