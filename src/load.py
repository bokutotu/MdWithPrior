import os
from pathlib import Path
from glob import glob
from urllib.parse import urlparse
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
import mlflow

from src.setup import setup_model
from src.experiment import Experiment


def get_artifact(run_id):
    uri = mlflow.get_run(run_id).info.artifact_uri
    path = urlparse(uri).path
    local_path = "/".join(path.split("/")[0:-1])
    return local_path


def load_from_run_id(run_id):
    artifacts_path = get_artifact(run_id)
    config_path = os.path.join(artifacts_path, "artifacts/config.yaml")
    model_path = os.path.join(artifacts_path, "artifacts/model.pth")
    config = OmegaConf.load(config_path)
    state_dict = torch.load(model_path)
    model = setup_model(config)
    model.load_state_dict(state_dict)
    return model
