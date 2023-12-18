from pathlib import Path
from typing import Any

import yaml
import numpy as np

from config.config import HPARAMS_ROOT
from src.main.activation import activation_dict
from src.main.initializer import initializer_dict
from src.main.loss import loss_dict
from src.main.metric import metrics_dict
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD


def read_yaml(path: str | Path) -> dict[str, Any]:
    """
    Reads a file in .yaml format.

    :param path: the path of the file to read
    :return: the dictionary contained in the file
    """
    with open(path, "r") as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    return dictionary


def load_hparams(model: str) -> dict[str, Any]:
    """
    Loads the hyperparameters for a certain model.

    :param model: the name of the model
    :return: the hyperparameters for the given model
    """
    return read_yaml(HPARAMS_ROOT.format(model))


def shuffle_data(X, y):
    """shuffle data inside the dataset"""
    num_samples = X.shape[0]
    shuffled_indices = np.random.permutation(num_samples)

    return X[shuffled_indices], y[shuffled_indices]


def create_model(
        units: list[int],
        activations: list[str],
        initializer: str,
        learning_rate: float,
        momentum: float,
        loss: str,
        metrics: list[str],
        **_
):
    assert len(units) - 1 == len(activations)
    model = Model()
    for i in range(len(units) - 1):
        activation = activation_dict.get(activations[i])
        initializer = initializer_dict.get(initializer)
        print(initializer)
        model.add(Dense(units[i], units[i + 1], initializer, (-0.7, 0.7), activation))

    sgd = SGD(learning_rate, momentum)
    loss = loss_dict.get(loss)
    metrics = [metrics_dict.get(metric) for metric in metrics]
    model.compile(sgd, loss, metrics)


def initialize_score(model):  # FORSE INUTILE(?)
    """initialize all the scores of the model to -1"""
    model_score = {"loss": float('inf')}
    for metric in model.metrics:
        model_score[metric.to_string()] = -1

    return model_score


def mean_std_scores(scores):
    """compute the mean and the std of the scores returned as dictionaries"""

    mean_score = {}
    std_score = {}
    keys = scores[0].keys()
    scores = list(map(list, zip(*[[entry[key] for key in entry] for entry in scores])))
    result_dict = {key: values for key, values in zip(keys, scores)}

    for keys in result_dict:
        mean_score[keys] = np.array(result_dict[keys]).mean()
        std_score[keys] = np.array(result_dict[keys]).std()

    return mean_score, std_score
