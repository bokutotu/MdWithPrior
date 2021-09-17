import multiprocessing

import numpy as np
import torch
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch import optim

from hydra.utils import instantiate

from src.model import CGnet
from src.dataset import MLPDataset, LSTMDataset
from src.statics import get_statics


def setup_dataloader(cfg, coordinates_path, forces_path,
                     train_test_rate, batch_size):
    """Set up dataloader
    Split the data from the npy file given by cfg with train, validation,
    and test, and create and return a data loader for each.

    Parameters
    ----------
    cfg: OmegaConf
        cfg about dataset
        cfg.dataset
    coordinates_path: str
        path to coordinates npy file
    forces_path: str
        path to forces npy file
    train_test_rate: float
        train data rate

    Returns
    -------
    train_dataloader: torch.DataLoader
    val_dataloader: torch.DataLoader
    test_dataloader: torch.DataLoader
    """
    coordinates = np.load(coordinates_path)
    forces = np.load(forces_path)
    len_coord = len(coordinates)

    train_last_idx = int(pow(train_test_rate, 2) * len_coord)
    val_last_idx = int(train_test_rate * len_coord)

    train_coord = coordinates[0:train_last_idx]
    train_force = forces[0: train_last_idx]
    val_coord = coordinates[train_last_idx: val_last_idx]
    val_force = forces[train_last_idx: val_last_idx]
    test_coord = coordinates[val_last_idx:-1]
    test_force = forces[val_last_idx: -1]

    train_dataset = instantiate(
        cfg, coordinates=train_coord, forces=train_force)
    val_dataset = instantiate(cfg, coordinates=val_coord, forces=val_force)
    test_dataset = instantiate(cfg, coordinates=test_coord, forces=test_force)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def setup_model(cfg):
    """get CGnet

    Parameters
    ----------
    cfg: OmegaConf
        config for CGnet
    coordinates_path: str
        path string for coordinates npy file
    forces_path: str
        path string for forces npy file

    Returns
    -------
    CGnet: torch.Modeule
    """
    coordinates = np.load(cfg.coordinates_path)
    stat = get_statics(torch.tensor(coordinates))
    num_atom = coordinates.shape[1]

    del coordinates

    net = CGnet(
        config=cfg, num_atom=num_atom,
        angle_mean=stat["angle"]["mean"],
        angle_std=stat["angle"]["std"],
        dihedral_mean=stat["dihedral"]["mean"],
        dihedral_std=stat["dihedral"]["std"],
        length_mean=stat["length"]["mean"],
        length_std=stat["length"]["std"])
    return net
