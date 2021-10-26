from urllib.parse import urlparse

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np
import mlflow
import shutil

from src.setup import setup_dataloader, setup_model


def epoch_step(model, dataloader, loss_func, optimizer, device, is_train=True):
    """1 epoch step

    Parameters
    ---------
    model: nn.Module
        train model
    dataloader: DataLoader
        dataloder
    loss_func: nn.Module
    optimizer: Optim
    device: cpu or GPU
    is_train: bool
        optim model or not
    """
    loss_sum = 0.0

    for (idx, batch) in enumerate(dataloader):
        x, y = batch
        x = x.to(device).requires_grad_(True)
        y = y.to(device).requires_grad_(True)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = loss_func(out, y)

        if is_train:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.detach().cpu())

    loss = loss_sum / len(dataloader)
    return model, loss


def train(cfg):
    """Train model function

    Parameters
    ----------
    cfg: DictConfig
    """
    print("set up mlflow experiment")
    mlflow.set_tracking_uri(
        'file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run() as run:
        print("set up dataloader")
        train_dataloader, val_dataloader, test_dataloader = setup_dataloader(
            cfg.dataset, cfg.coordinates_path, cfg.forces_path,
            cfg.train_test_rate, cfg.batch_size
        )
        print("len trian dataloader is {}".format(len((train_dataloader))))

        print("set up model")
        model = setup_model(cfg)
        optimizer = instantiate(cfg.optimizer, params=model.parameters(), )
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
        loss_func = torch.nn.MSELoss()
        print(model)

        device = torch.device(
                "cuda:{}".format(cfg.gpus[0])) if cfg.gpus is not None else torch.device("cpu")
        model = model.to(device)

        train_loss = []
        val_loss = []

        best_model_parameters = None
        best_loss = 1e-10

        parameters_list = []

        print("start train")
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
            print("epoch: {} train: {:.4f}, val: {:.4f}"
                  .format(i+1, train_loss[-1], val_loss[-1]))

            mlflow.log_metrics(
                {"train loss": train_loss[-1], "validation loss": val_loss[-1]}, i)

            scheduler.step()

            # Keep the parameters of the model with the best validation value.
            if best_loss < val_loss[-1]:
                best_model_parameters = model.state_dict()

        # test use best model
        model.load_state_dict(best_model_parameters)
        _, loss = epoch_step(
            model, test_dataloader, loss_func, optimizer, device, False)

        print("test loss {:.4f}".format(loss))

        # save best model to artifact dir
        uri = run.info.artifact_uri
        path = urlparse(uri).path
        torch.save(best_model_parameters, path + "/model.pth")

        # save best model to mlruns dir
        sorted_arg = np.argsort(val_loss).tolist()
        best_model_idx = sorted_arg[0]

        # save config to mlruns dir
        OmegaConf.save(cfg, path + "/config.yaml")

        mlflow.log_metric("test loss", loss)
