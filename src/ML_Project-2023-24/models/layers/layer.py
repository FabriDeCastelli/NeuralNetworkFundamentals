
import numpy as np


class Layer:
    """
    Abstract class representing a layer in a neural network.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer.
        Has to be called on subclasses.

        :param x: input to the layer
        :return: output of the layer
        """
        raise NotImplementedError()

    def backward(self, error: np.ndarray):
        """
        Performs a backward pass of the layer.
        Has to be called on subclasses.

        :param error: error to be propagated back
        :return: error to prop to the previous layer
        """
        raise NotImplementedError()
