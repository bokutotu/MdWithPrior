import hydra
from omegaconf import DictConfig
from mlflow.pytorch import save_model

# from src.experiment import Experiment
from src.train import train


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    # exp = Experiment(config)
    # exp.run()
    # exp.save_output()
    # save_model(exp.model, "model")
    train(config)


if __name__ == "__main__":
    main()
