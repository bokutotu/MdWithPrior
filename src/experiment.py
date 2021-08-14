import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from src.data_module import DataModule
from src.model import CGnet
from torch import Tensor
from torch.optim import Optimizer


class Experiment(pl.LightningModule):
    def __init__(self, config, angle_mean, angle_std,
                 length_mean, length_std, dihedral_mean, dihedral_std):
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

        self.data_module = DataModule(
            config.batch_size, config.coordinates_path, config.force_path,
            config.dataset, config.train_test_rate, config.dataset)

        self.model = CGnet(config=config, num_atom=num_atom,
                           angle_mean=angle_mean, angle_std=angle_std,
                           dihedral_mean=dihedral_mean, dihedral_std=dihedral_std,
                           length_mean=length_mean, length_std=length_std)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(
            self.config.optimizer, params=params)
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def loss_fn(self, output, ans):
        loss = F.mse_loss(output, ans)
        return loss

    def training_step(self, batch, batch_idx):
        input, ans = batch
        force, energy = self.model(input)
        loss = self.loss_fn(force, ans)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        input, ans = batch
        force, energy = self.model(input)
        loss = self.loss_fn(force, ans)
        self.log("val_loss", loss)
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

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
