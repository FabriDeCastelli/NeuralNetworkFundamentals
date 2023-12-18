import numpy as np


class Initializer:

    def __init__(self):
        pass

    def weight_init(self, *_):
        """
        Initializes weights of a layer according to a heuristic

        :param : the shape of the weight matrix as number of inputs and outputs of the layer
        :return: the initialized weight matrix
        """

        raise NotImplementedError()

    def bias_init(self, *_):
        """
        Initializes biases of a layer according to a heuristic

        :param _: the number of neurons of the layer
        """
        raise NotImplementedError()


class Random(Initializer):
    """
    Initializes weights and biases randomly
    """

    def __init__(self):
        super().__init__()

    def weight_init(self, shape, *_):
        return np.random.randn(*shape)

    def bias_init(self, shape):
        return np.random.randn(*shape)


class Range(Initializer):
    """
    Initializes weights and biases randomly
    """

    def __init__(self):
        super().__init__()

    def weight_init(self, shape, range: (float, float)):
        min = range[0]
        max = range[1]
        return np.random.uniform(min, max, shape)

    def bias_init(self, shape):
        return np.zeros(shape)


class FanIn(Initializer):
    """
    Initializes weights and biases randomly
    """

    def __init__(self):
        super().__init__()

    def weight_init(self, shape, *_):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])

    def bias_init(self, shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])


initializer_dict = {
    "random": Random(),
    "range": Range(),
    "fan_in": FanIn()
}