import numpy as np

from src.main.evaluation.grid_search import GridSearch
from src.main.utilities.utils import shuffle_data
from sklearn.model_selection import train_test_split


def holdout_CV(X: np.ndarray,
               y: np.ndarray,
               grid_search: GridSearch,
               split: float = 0.3,
               verbose: bool = False
               ):
    """
    function that allow to compute the double kfold cross validation
    
    :param X: dataset to train the model
    :param y: target of the dataset
    :param k: number of fold
    :param grid_search: a grid search object
    :param verbose: verbose mode for fit function
    :return: mean and std of train, validation and test scores as dictionary, best model
    """

    train_mean, train_std, val_mean, val_std, test_score, test_std = {}, {}, {}, {}, {}, {}

    X, y = shuffle_data(X, y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=1)

    """train_mean, train_std, val_mean, val_std,"""
    ((train_mean, train_std), (val_mean, val_std)), model, params, histories = grid_search.run_search(x_train, y_train,
                                                                                                      verbose)
    _, _ = params

    """print("------- BEFORE REFIT -------")

    model.initialize_weights()
    
    print("TRAIN MEAN:", train_mean[model.get_loss().to_string()])
    print("TEST SCORE:" , model.evaluate(x_test, y_test))

    print("------- AFTER REFIT -------")
    model, history = model.refit(x_train , y_train, x_test, y_test, train_mean, 0.00001, batch_size, True)
    """
    test_score = model.evaluate(x_test, y_test)

    for key in test_score.keys():
        test_std[key] = 0.0

    return train_mean, train_std, val_mean, val_std, test_score, test_std, model, histories
