import os
from urllib.parse import urlparse
import functools

import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import mlflow

from src.statics import get_statics
from src.data_module import DataModule
from src.setup import setup_model, setup_dataloader
from src.model import CGnet


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

        self.loss_fn = torch.nn.MSELoss()

        print(self.model)

        self.val_loss = 1e-20
        self.best_model_state_dict = self.model.state_dict()

        self.tensor_dtype = torch.float32 if config.trainer.precision == 32 else torch.float16

        # del coordinates

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(
            self.config.optimizer, params=params)
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    # def loss_fn(self, output, ans):
    #     loss = F.mse_loss(output, ans)
    #     return loss

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.requires_grad_(True)
        y = x.requires_grad_(True)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = loss_func(out, y)
        return loss

    def training_epoch_end(self, loss):
        loss = [float(item["loss"].detach().cpu()) for item in loss]
        loss_sum = functools.reduce(lambda a, b: a + b, loss)
        loss_avg = loss_sum / len(loss)
        self.log("train_loss", loss_avg)

    @torch.enable_grad()
    def validation_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = loss_func(out, y)
        return loss

    def validation_epoch_end(self, loss):
        loss = [float(i.detach().cpu()) for i in loss]
        loss_sum = functools.reduce(lambda a, b: a + b, loss)
        loss_avg = loss_sum / len(loss)

        if loss_avg <= self.val_loss:
            self.val_loss = loss_avg
            self.best_model_state_dict = self.model.state_dict()
        self.log("validation_loss", loss_avg)

    @torch.enable_grad()
    def test_step(self, batch: Tensor, batch_idx: int):
        x, y = batch
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = loss_func(out, y)
        return loss

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
