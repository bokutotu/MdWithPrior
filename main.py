import hydra
from omegaconf import DictConfig
from mlflow.pytorch import save_model
from src.train import train
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# from src.experiment import Experiment


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    # exp = Experiment(config)
    # exp.run()
    # exp.save_output()
    # save_model(exp.model, "model")
    train(config)


if __name__ == "__main__":
    main()
