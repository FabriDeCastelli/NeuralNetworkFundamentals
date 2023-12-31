from pathlib import Path
from typing import Any

import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from config.config import HPARAMS_ROOT, PROJECT_FOLDER_PATH
from src.main.activation import activation_dict
from src.main.callback import Callback, EarlyStopping
from src.main.initializer import Initializer
from src.main.loss import loss_dict
from src.main.metric import metrics_dict
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
from src.main.regularizer import Regularizer, regularizer_dict


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
        loss: str,
        metrics: list[str],
        restore_best_weights,
        monitor,
        delta,
        start_from_epoch,
        patience,
        regularizer: str | Regularizer = None,
        weight_initializer: str | Initializer = 'glorot_uniform',
        bias_initializer: str | Initializer = 'zeros',
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        lambd: float = 0.0,
        **_
):
    assert len(units) - 1 == len(activations)
    assert loss
    assert metrics

    model = Model()
    for i in range(len(units) - 1):
        model.add(Dense(units[i], units[i + 1], weight_initializer, bias_initializer, activations[i]))

    sgd = SGD(learning_rate, momentum)
    regularizer = regularizer_dict.get(regularizer)

    if regularizer:
        regularizer.set_lambda(lambd)

    callback = EarlyStopping(patience, start_from_epoch, delta, monitor, restore_best_weights)
    model.compile(sgd, loss, metrics, callback, regularizer)

    return model


def initialize_score(model):
    """initialize all the scores of the model to -1"""
    model_score = {"loss": float('inf')}
    for metric in model.metrics:
        model_score[repr(metric)] = -1

    return model_score


def mean_std_scores(scores):
    """compute the mean and the std of the scores returned as dictionaries
    
    :param scores: list of dictionaries
    """

    mean_score = {}
    std_score = {}
    keys = scores[0].keys()
    scores = list(map(list, zip(*[[entry[key] for key in entry] for entry in scores])))
    result_dict = {key: values for key, values in zip(keys, scores)}

    for keys in result_dict:
        mean_score[keys] = np.array(result_dict[keys]).mean()
        std_score[keys] = np.array(result_dict[keys]).std()

    return mean_score, std_score


def compute_metrics(model, train_mean, train_std, val_mean, val_std, test_mean, test_std):
    """
    take the model and the scores on the set and create a pandas dataframe with the results
    """

    def round_number(x, y, z):
        return [round(x, 4), round(y, 4), round(z, 4)]

    means = []
    stds = []
    sets = ["Training", "Validation", "Test"]

    metrics = [repr(metric) for metric in model.get_metrics() + [model.get_loss()]]

    for metric in metrics:
        means.extend(round_number(train_mean[metric], val_mean[metric], test_mean[metric]))
        stds.extend(round_number(train_std[metric], val_std[metric], test_std[metric]))

    metrics = [el for el in metrics for _ in range(3)]

    data = {'Metrics': metrics, 'Set': sets * (len(metrics) // 3), 'Mean': means, 'Std': stds}
    df = pd.DataFrame(data)
    return df


def log_experiment(exp_dir, model, train_mean, train_std, val_mean, val_std, test_mean, test_std, histories=None):
    """
    Logs the results of an experiments on a csv file inside exp_dir

    :param exp_dir: the path of the experiment directory
    :param model: the model used for the experiment
    :param train_mean: the mean of the scores on the training set
    :param train_std: the standard deviation of the scores on the training set
    :param val_mean: the mean of the scores on the validation set
    :param val_std: the standard deviation of the scores on the validation set
    :param test_mean: the mean of the scores on the test set
    :param test_std: the standard deviation of the scores on the test set
    """
    metrics = compute_metrics(model, train_mean, train_std, val_mean, val_std, test_mean, test_std)
    metrics.to_csv(exp_dir / "metrics.csv", index=False, decimal=',')
    if histories is not None:
        for i,history in enumerate(histories):
            if len(histories) == 1:
                fold_dir = exp_dir / f"monk_plot"
                fold_dir.mkdir(exist_ok=True, parents=True)
            else:
                fold_dir = exp_dir / f"fold_{i+1}"
                fold_dir.mkdir(exist_ok=True, parents=True)
            plot_history(history, fold_dir)

    model.save(exp_dir / "model.json")
            

def setup_experiment(name: str) -> Path:
    """
    Initializes experiment by creating the proper directory

    :param name: the name of the experiment as string
    :return: the path of the experiment directory
    """
    root = PROJECT_FOLDER_PATH / Path("experiments")
    exp_dir = root / name
    exp_dir.mkdir(exist_ok=True, parents=True)
    return exp_dir


def plot_history(history, exp_dir = None):
    """
    plot the history of the training and the test set
    
    :param history: the history of the training and test set
    """
    sns.set_context("notebook")
    sns.set_theme(style="whitegrid")

    for metric in history.keys():
        epochs = np.arange(1, len(history[metric]['training']) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[metric]['training'], label='Training')
        if "test" in history[metric].keys():
            plt.plot(epochs, history[metric]['test'], label='Test') #, color='#C80000', linestyle='--') 
        else:
            plt.plot(epochs, history[metric]['validation'], label='Validation') #, color='#C80000', linestyle='--') 
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(f'{metric.capitalize()} Value')
        plt.legend()
        if exp_dir is not None:
            plt.savefig(exp_dir / f'{metric}.pdf')
        #plt.show()
