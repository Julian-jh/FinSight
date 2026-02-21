import yaml
import joblib
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_model(model: Any, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
