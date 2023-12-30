import numpy as np


class Initializer:

    def __call__(self, shape):
        """
        Initializes weights of a layer according to a heuristic

        :param shape: the shape of the array to be initialized. Can be either the bias or the weights.
        :return: the initialized array
        """
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__


class RandomNormal(Initializer):
    """
    Initializes weights and biases randomly, according to the normal distribution
    """

    def __init__(self, mean=0.0, std=0.01):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        return np.random.normal(self.mean, self.std, shape)


class Range(Initializer):
    """
    Initializes weights and biases randomly, according to the uniform distribution, in a given range
    """

    def __init__(self, range=(-0.2, 0.2)):
        super().__init__()
        self.range = range

    def __call__(self, shape):
        min_val, max_val = self.range
        return np.random.uniform(min_val, max_val, shape)


class FanIn(Initializer):
    """
    Fan-in initializer.
    """

    def __call__(self, shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)


class Zeros(Initializer):
    """
    Zeros initializer.
    """

    def __call__(self, shape):
        return np.zeros(shape)


class GlorotUniform(Initializer):
    """
    Glorot Uniform initializer.
    """

    def __call__(self, shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)


initializer_dict = {
    "random_normal": RandomNormal(),
    "range": Range(),
    "fan_in": FanIn(),
    "zeros": Zeros(),
    "glorot_uniform": GlorotUniform()
}
