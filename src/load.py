from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.setup import setup_model


def load_from_run_id(run_id):
    current_dir = Path.cwd()
    artifact_dir = current_dir / "mlruns/1/{}/artifacts".format(run_id)
    config_yaml = OmegaConf.load(str(artifact_dir / "config.yaml"))
    model = setup_model(config_yaml)
    state_dict = torch.load(str(artifact_dir / "model.pth"))
    model.load_state_dict(state_dict)
    return model
