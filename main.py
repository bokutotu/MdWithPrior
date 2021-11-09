from src.experiment import Experiment
import hydra
from omegaconf import DictConfig
# from src.train import train

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    exp = Experiment(config)
    exp.run()
    # train(config)


if __name__ == "__main__":
    main()
