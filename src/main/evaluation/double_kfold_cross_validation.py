import numpy as np

from .grid_search import GridSearch
from src.main.utils import shuffle_data


def double_Kfold_CV(X, y, grid_search="grid_search", k=5, compile_params=None):
    """
    function that allow to compute the double kflod cross validation
    
    :param X: dataset to train the model
    :param y: target of the dataset
    :param k: number of fold
    :param model: the model to be trained
    :param grid_search: a grid search string that allow to choose the grid search to be used
    :param compile_params: compile parameters for the model
    """

    X, y = shuffle_data(X, y)

    x_fold = np.array_split(X, k)
    y_fold = np.array_split(y, k)

    grid_search = GridSearch.create_search_object(grid_search)

    train_scores = []
    test_scores = []

    for i in range(k):
        x_train, x_test = x_fold[:i] + x_fold[i + 1:], x_fold[i]
        y_train, y_test = y_fold[:i] + y_fold[i + 1:], y_fold[i]

        model = grid_search.run_seach(x_train, y_train, x_test, y_test, compile_params)

    pass
