from pathlib import Path
from typing import Any
import yaml

def read_yaml(path: str | Path) -> dict[str, Any]:
    """Reads a file in .yaml format."""
    with open(path, "r") as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    return dictionary

def load_hparams(classifier: str) -> dict[str, Any]:
    """Loads the hyperparameters for a certain model."""
    return read_yaml(HPARAMS_ROOT / f"{classifier}.yaml")