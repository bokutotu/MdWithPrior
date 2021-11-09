import os
import multiprocessing
from typing import Optional

import numpy as np
from hydra.utils import instantiate
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from src.dataset import MLPDataset, LSTMDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, coordinates_path,
                 forces_path, train_test_rate, config,norm):
        super().__init__()

        coordinates = np.load(coordinates_path)
        forces = np.load(forces_path)
        len_coord = len(coordinates)

        train_last_idx = int(pow(train_test_rate, 2) * len_coord)
        val_last_idx = int(train_test_rate * len_coord)

        self.train_coord = coordinates[0:train_last_idx]
        self.train_force = forces[0: train_last_idx]
        self.val_coord = coordinates[train_last_idx: val_last_idx]
        self.val_force = forces[train_last_idx: val_last_idx]
        self.test_coord = coordinates[val_last_idx:-1]
        self.test_force = forces[val_last_idx: -1]

        self.config = config
        self.batch_size = batch_size
        self.norm = norm
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # make assignments here (val/train/test split)
        # called on every GPUs
        self.train = instantiate(
            self.config, coordinates=self.train_coord, forces=self.train_force, 
            norm=self.norm)
        self.val = instantiate(
            self.config, coordinates=self.val_coord, forces=self.val_force, 
            norm=self.norm)
        self.test = instantiate(
            self.config, coordinates=self.test_coord, forces=self.test_force, 
            norm=self.norm)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            # sampler=RandomSampler(self.train),
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            # sampler=RandomSampler(self.val),
            num_workers=multiprocessing.cpu_count(),
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            # sampler=RandomSampler(self.test),
            num_workers=multiprocessing.cpu_count(),
            shuffle=False
        )

    def teardown(self, stage: Optional[str] = None):
        # clean up after fit or test
        # called on every process in DDP
        pass
