import numpy as np
from config.config import *


def get_cup_training_set() -> (np.ndarray, np.ndarray):
    """
    Reads the training set from the csv file and returns it as a numpy array

    :return: the training set as a numpy array
    """

    data = np.genfromtxt(ml_cup_path + "/training.csv", delimiter=';', skip_header=1)[:, 1:]

    features = data[:, :-3]
    labels = data[:, -3:]

    return features, labels


def get_cup_test_set() -> np.ndarray:
    """
    Reads the blind test set from the csv file and returns it as a numpy array

    :return: the test set as a numpy array
    """

    data = np.genfromtxt(ml_cup_path + "/test.csv", delimiter=';', skip_header=1)[:, 1:]

    test = data[:, :-3]

    return test


def get_monk_training_set(number: int) -> (np.ndarray, np.ndarray):
    """
    Reads the training set from the csv file and returns it as a numpy array

    :param number: the number of the monk benchmark
    :return: the training features and labels as two numpy arrays
    """

    training_set_path = monk_benchmark_path.format(number) + "/monks-{}.train".format(number)
    training_set = np.genfromtxt(training_set_path, delimiter=' ', skip_header=1)
    features = training_set[:, 1:-1]
    labels = training_set[:, 0]

    return features, labels


def get_monk_test_set(number: int) -> (np.ndarray, np.ndarray):
    """
    Reads the test set from the csv file and returns it as a numpy array

    :param number: the number of the monk benchmark
    :return: the test features and labels as two numpy arrays
    """

    test_set_path = monk_benchmark_path.format(number) + "/monks-{}.test".format(number)
    test = np.genfromtxt(test_set_path, delimiter=' ', skip_header=1)
    features = test[:, 1:-1]
    labels = test[:, 0]

    return features, labels

