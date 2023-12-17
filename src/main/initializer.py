import numpy as np


class Initializer:

    @staticmethod
    def weight_init(*_):
        """
        Initializes weights of a layer according to a heuristic

        :param _: the shape of the weight matrix as number of inputs and outputs of the layer
        :return: the initialized weight matrix
        """

        raise NotImplementedError()

    @staticmethod
    def bias_init(*_):
        """
        Initializes biases of a layer according to a heuristic

        :param _: the number of neurons of the layer
        """
        raise NotImplementedError()


class Random(Initializer):
    """
    Initializes weights and biases randomly
    """

    @staticmethod
    def weight_init(shape, *_):
        return np.random.randn(*shape)

    @staticmethod
    def bias_init(shape):
        return np.random.randn(*shape)


class Range(Initializer):
    """
    Initializes weights and biases randomly
    """

    @staticmethod
    def weight_init(shape, range: (float, float)):
        min = range[0]
        max = range[1]
        return np.random.uniform(min, max, shape)

    @staticmethod
    def bias_init(shape):
        return np.zeros(shape)


class FanIn(Initializer):
    """
    Initializes weights and biases randomly
    """

    @staticmethod
    def weight_init(shape, *_):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])

    @staticmethod
    def bias_init(shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])
