import numpy as np

class Initializer:

    def weight_init(self, _):
        """
        Initializes weights of a layer according to a heuristic

        :param _: the shape of the weight matrix as number of inputs and outputs of the layer
        :return:
        """

        raise NotImplementedError()

    def bias_init(self, _):
        """
        Initializes biases of a layer according to a heuristic
        :param _: the number of neurons of the layer
        :return:
        """
        raise NotImplementedError()


class Random(Initializer):
    """
    Initializes weights and biases randomly
    """

    def weight_init(self, shape):
        return np.random.randn(*shape)

    def bias_init(self, shape):
        return np.random.randn(*shape)


class Range(Initializer):
    """
    Initializes weights and biases randomly
    """

    def weight_init(self, shape, min, max):
        return np.random.uniform(min, max, shape)

    def bias_init(self, shape):
        return np.zeros(shape)
    

class FanIn(Initializer):
    """
    Initializes weights and biases randomly
    """

    def weight_init(self, shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])

    def bias_init(self, shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])