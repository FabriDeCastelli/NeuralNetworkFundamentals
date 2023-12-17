import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config.config import *


def get_cup_training_set() -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Reads the training set from the csv file and returns it as a numpy array

    :return: the training set as a numpy array
    """

    data = np.genfromtxt(ml_cup_path + "/training.csv", delimiter=';', skip_header=1)[:, 1:]

    x_train = data[:, :-3]
    y_train = data[:, -3:]

    x_test = np.genfromtxt(ml_cup_path + "/test.csv", delimiter=';', skip_header=1)[:, 1:]

    assert x_train.shape[1] == x_test.shape[1]

    return x_train, y_train, x_test


def get_monk(number: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Reads a monk benchmark from the csv file and returns the training and test set with their labels as numpy arrays

    :param number: the number of the monk benchmark to read
    :return: the training and test set together with their labels as numpy arrays
    """
    training_set_path = monk_benchmark_path.format(number) + "/monks-{}.train".format(number)
    test_set_path = monk_benchmark_path.format(number) + "/monks-{}.test".format(number)

    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']

    # read training set
    x_train = pd.read_csv(training_set_path, sep=' ', names=col_names)
    x_train.set_index('Id', inplace=True)
    y_train = x_train.pop('class')

    # read test set
    x_test = pd.read_csv(test_set_path, sep=' ', names=col_names)
    x_test.set_index('Id', inplace=True)
    y_test = x_test.pop('class')

    # 1-hot encoding (and transform dataframe to numpy array)
    x_train = OneHotEncoder().fit_transform(x_train).toarray().astype(np.float32)
    y_train = OneHotEncoder().fit_transform(y_train.to_numpy()[:, np.newaxis]).toarray().astype(np.float32)
    x_test = OneHotEncoder().fit_transform(x_test).toarray().astype(np.float32)
    y_test = OneHotEncoder().fit_transform(y_test.to_numpy()[:, np.newaxis]).toarray().astype(np.float32)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    assert x_train.shape[1] == x_test.shape[1]
    assert y_train.shape[1] == y_test.shape[1]

    return x_train, y_train, x_test, y_test
