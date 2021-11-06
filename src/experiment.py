import os
from urllib.parse import urlparse

import torch
from torch import Tensor
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

from src.data_module import DataModule
from src.setup import setup_model


class Experiment(pl.LightningModule):
    def __init__(self, config, ):
        super(Experiment, self).__init__()
        self.config: DictConfig = config
        logger = instantiate(config.logger)
        self.trainer = instantiate(
            config.trainer,
            logger=logger,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
            ],
        )

        self.model = setup_model(config)

        self.data_module = DataModule(
            config.batch_size, config.coordinates_path, config.forces_path,
            config.train_test_rate, config.dataset)

        self.loss_func = torch.nn.MSELoss(reduction="mean")

        print(self.model)

        self.val_loss = 1e-20
        self.best_model_state_dict = self.model.state_dict()

        self.tensor_dtype = torch.float32 if config.trainer.precision == 32 else torch.float16

        self.warm_up = config.warm_up
        self.norm = config.norm

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(
            self.config.optimizer, params=params)
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def cal_nn(self, x):
        is_use_NN = self.current_epoch >= self.warm_up
        return self.model(x, is_use_NN)

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        out, _ = self.cal_nn(x)
        loss = self.loss_func(out, y)
        return loss

    def training_epoch_end(self, loss):
        loss = np.array([float(item["loss"].detach().cpu()) for item in loss])
        loss_avg = loss.mean()
        self.log("train_loss", loss_avg)

    @torch.enable_grad()
    def validation_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        out, _ = self.cal_nn(x)
        loss = self.loss_func(out, y)
        return loss

    def validation_epoch_end(self, loss):
        loss = np.array([float(i.detach().cpu()) for i in loss])
        loss_avg = loss.mean()

        if loss_avg <= self.val_loss:
            self.val_loss = loss_avg
            self.best_model_state_dict = self.model.state_dict()
        self.log("validation_loss", loss_avg)

    @torch.enable_grad()
    def test_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        out, _ = self.model(x)
        loss = self.loss_func(out, y)
        return loss

    def test_epoch_end(self, loss):
        loss = np.array([float(i.detach().cpu()) for i in loss])
        loss_avg = loss.mean()

        self.log("test_loss", loss_avg)

    # train your model
    def fit(self):
        self.trainer.fit(self, self.data_module)
        self.logger.log_hyperparams(
            {
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
            }
        )
        self.log_artifact(".hydra/config.yaml")
        self.log_artifact(".hydra/hydra.yaml")
        self.log_artifact(".hydra/overrides.yaml")
        self.log_artifact("main.log")

    # run your whole experiments
    def run(self):
        artifact_path = urlparse(self.logger._tracking_uri).path
        artifact_path = os.path.join(
            artifact_path, self.logger.experiment_id, self.logger.run_id, "artifacts")
        self.fit()
        self.trainer.test()
        torch.save(self.best_model_state_dict, artifact_path + "/model.pth")

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
