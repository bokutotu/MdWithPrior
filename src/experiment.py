import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

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

        del coordinates

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(
            self.config.optimizer, params=params)
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def loss_fn(self, output, ans):
        # loss = F.mse_loss(output, ans)
        loss = ForceLoss()(output, ans)
        return loss

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        input, ans = batch
        input = torch.tensor(input, requires_grad=True, dtype=torch.float32)
        ans = torch.tensor(ans, requires_grad=True, dtype=torch.float32)
        force, energy = self.model(input)
        loss = self.loss_fn(force, ans)
        self.log("train_loss", loss)
        return loss

    @torch.enable_grad()
    def validation_step(self, batch: Tensor, batch_idx: int):
        input, ans = batch
        input = torch.tensor(input.cpu().numpy(), requires_grad=True)
        ans = torch.tensor(ans.cpu().numpy(), requires_grad=True)
        force, energy = self.model(input)
        loss = self.loss_fn(force, ans)
        self.log("val_loss", loss)
        return loss

    @torch.enable_grad()
    def test_step(self, batch: Tensor, batch_idx: int):
        input, ans = batch
        input = torch.tensor(input.cpu().numpy(), requires_grad=True)
        ans = torch.tensor(ans.cpu().numpy(), requires_grad=True)
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
        self.fit()
        self.trainer.test()

    def save_output(self):
        coord_npy = np.load(self.config.coordinates_path)
        coord_tensor = torch.tensor(coord_npy, requires_grad=True)
        output_tensor, _ = self.model(coord_tensor)
        output_npy = output_tensor.detach().numpy()
        np.save("/tmp/my_impl.npy", output_npy)

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
