import numpy as np

from src.main.evaluation.grid_search import GridSearch
from src.main.utilities.utils import shuffle_data
from sklearn.model_selection import train_test_split


def holdout_CV(X: np.ndarray,
               y: np.ndarray,
               grid_search: GridSearch,
               split: float = 0.1,
               verbose: bool = False
               ):
    """
    function that allow to compute the double kfold cross validation

    :param X: dataset to train the model
    :param y: target of the dataset
    :param grid_search: a grid search object
    :param split: training-test set split ratio
    :param verbose: verbose mode for fit function
    :return: mean and std of train, validation and test scores as dictionary, best model
    """

    train_mean, train_std, val_mean, val_std, test_score, test_std = {}, {}, {}, {}, {}, {}

    X, y = shuffle_data(X, y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=1)

    ((train_mean, train_std), (val_mean, val_std)), model, params, histories, top5 = (
        grid_search.run_search(x_train, y_train, verbose=verbose)
    )

    test_score = model.evaluate(x_test, y_test)

    for key in test_score.keys():
        test_std[key] = 0.0

    return train_mean, train_std, val_mean, val_std, test_score, test_std, model, params, histories, top5
