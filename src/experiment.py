import os
from urllib.parse import urlparse

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

        coordinates = np.load(config.coordinates_path)
        coordinates = torch.tensor(coordinates)
        stat = get_statics(coordinates)

        self.data_module = DataModule(
            config.batch_size, config.coordinates_path, config.forces_path,
            config.train_test_rate, config.dataset)

        num_atom = coordinates.shape[1]

        self.model = CGnet(config=config, num_atom=num_atom,
                           angle_mean=stat["angle"]["mean"],
                           angle_std=stat["angle"]["std"],
                           dihedral_mean=stat["dihedral"]["mean"],
                           dihedral_std=stat["dihedral"]["std"],
                           length_mean=stat["length"]["mean"],
                           length_std=stat["length"]["std"])

        self.val_loss = 1e-20
        self.best_model_state_dict = self.model.state_dict()

        self.tensor_dtype = torch.float32 if config.trainer.precision == 32 else torch.float16

        del coordinates

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(
            self.config.optimizer, params=params)
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def loss_fn(self, output, ans):
        loss = F.mse_loss(output, ans)
        return loss

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        input, ans = batch
        input = torch.tensor(input, dtype=self.tensor_dtype,
                             requires_grad=True, device=torch.device("cuda"))
        ans = torch.tensor(ans, dtype=self.tensor_dtype,
                           requires_grad=True, device=torch.device("cuda"))
        force, energy = self.model(input)
        loss = self.loss_fn(force, ans)
        self.log("train_loss", loss)
        return loss

    @torch.enable_grad()
    def validation_step(self, batch: Tensor, batch_idx: int):
        input, ans = batch
        input = torch.tensor(input, dtype=self.tensor_dtype,
                             requires_grad=True, device=torch.device("cuda"))
        ans = torch.tensor(ans, dtype=self.tensor_dtype,
                           requires_grad=True, device=torch.device("cuda"))
        force, energy = self.model(input)
        loss = self.loss_fn(force, ans)
        self.log("val_loss", loss)
        loss = loss.detach().cpu()
        if loss <= self.val_loss:
            self.val_loss = loss
            self.best_model_state_dict = self.model.state_dict()
        return loss

    @torch.enable_grad()
    def test_step(self, batch: Tensor, batch_idx: int):
        input, ans = batch
        input = torch.tensor(input, dtype=self.tensor_dtype,
                             requires_grad=True, device=torch.device("cuda"))
        ans = torch.tensor(ans, dtype=self.tensor_dtype,
                           requires_grad=True, device=torch.device("cuda"))
        force, energy = self.model(input)
        loss = self.loss_fn(force, ans)
        self.log("test_loss", loss)
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
