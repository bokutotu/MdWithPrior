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
        return (self.coordinates[idx], self.forces[idx])


class LSTMDataset(Dataset):

    def __init__(self, coordinates, forces, features_length):
        self.coordinates = coordinates
        self.forces = forces
        self.features_length = features_length

    def __len__(self):
        return len(self.coordinates) - self.features_length

    def __getitem__(self, idx):
        input_array = self.coordinates[idx: idx+self.features_length]
        target_array = self.forces[idx: idx+self.features_length]
        return (input_array, target_array)
