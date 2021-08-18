from logging import config
import torch
from torch import optim
from torch.utils.data import DataLoader

from hydra.utils import instantiate
import numpy as np
import mlflow

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

    train_dataset = instantiate(cfg, coordinates=train_coord, forces=train_force)
    val_dataset = instantiate(cfg, coordinates=val_coord, forces=val_force)
    test_dataset = instantiate(cfg, coordinates=test_coord, forces=test_force)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def setup_model(cfg, coordinates_path, forces_path):
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
    coordinates = np.load(coordinates_path)
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


def epoch_step(model, dataloader, loss_func, optimizer, device, is_train=True):
    loss_sum = 0.0

    for (idx, batch) in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        y = x.to(device)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = loss_func(out, y)

        if is_train:
            loss.backward()
            optimizer.step()

        loss_sum += loss 

    loss = loss_sum / (idx + 1)
    return model, loss


def train(cfg):
    print("set up dataloader")
    train_dataloader, val_dataloader, test_dataloader = setup_dataloader(
        cfg.dataset, cfg.coordinates_path, cfg.forces_path, 
        cfg.train_test_rate, cfg.batch_size
    )

    print("set up model")
    model = setup_model(cfg, cfg.coordinates_path, cfg.forces_path)
    optimizer = instantiate(cfg.optimizer, params=model.parameters(), )
    loss_func = torch.nn.MSELoss()

    device = torch.device("cuda") if cfg.gpus is not None else torch.device("cpu")

    train_loss = []
    val_loss = []

    for i in range(cfg.epochs):
        # train
        model, loss = epoch_step(
            model, train_dataloader, loss_func, optimizer, device)
        train_loss.append(loss)

        # val
        _, loss = epoch_step(
            model, val_dataloader, loss_func, optimizer, device, False
        )
        val_loss.append(loss)
        print("epoch: {} train: {.4f}, val: {.4f}" \
            .format(i+1,train_loss[-1], val_loss[-1]))


    # test
    _, loss = epoch_step(
        model, test_dataloader, loss_func, optimizer, device, False
    )