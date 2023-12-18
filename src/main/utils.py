from pathlib import Path
from typing import Any

import yaml
import numpy as np

def read_yaml(path: str | Path) -> dict[str, Any]:
    """Reads a file in .yaml format."""
    with open(path, "r") as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    return dictionary

def load_hparams(classifier: str) -> dict[str, Any]:
    """Loads the hyperparameters for a certain model."""
    return read_yaml(HPARAMS_ROOT / f"{classifier}.yaml")

def shuffle_data(X, y):
    """shuffle data inside the dataset"""
    num_samples = X.shape[0]

    # Create shuffled indices
    shuffled_indices = np.random.permutation(num_samples)

    # Shuffle both arrays using the same shuffled indices

    return X[shuffled_indices], y[shuffled_indices]
