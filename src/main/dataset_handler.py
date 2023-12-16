import numpy as np
import pandas as pd


def get_training_set(file_path: str):
    """
    Reads the training set from the csv file and returns it as a numpy array

    :param file_path: the path to the csv file containing the training set
    :return: the training set as a numpy array
    """

    data = np.genfromtxt(file_path, delimiter=';', skip_header=1)[:, 1:]

    x_train = data[:, :-3]
    y_train = data[:, -3:]

    return x_train, y_train




