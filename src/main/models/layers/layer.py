
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

    def get_weights(self):
        """
        Returns the weights of the layer.
        """
        raise NotImplementedError()

    def get_bias(self):
        """
        Returns the bias of the layer.
        """
        raise NotImplementedError()

    def get_input(self):
        """
        Returns the input of the layer.
        """
        raise NotImplementedError()

    def get_delta_w_old(self):
        """
        Returns the delta_w_old of the layer.
        """
        raise NotImplementedError()

    def get_delta_b_old(self):
        """
        Returns the delta_b_old of the layer.
        """
        raise NotImplementedError()

    def set_delta_w_old(self, delta_w):
        """
        Sets the delta_w_old of the layer.
        """
        raise NotImplementedError()

    def set_delta_b_old(self, delta_b):
        """
        Sets the delta_b_old of the layer.
        """
        raise NotImplementedError()

    def set_weights(self, new_weights):
        """
        Sets the weights of the layer.
        """
        raise NotImplementedError()

    def set_bias(self, new_bias):
        """
        Sets the bias of the layer.
        """
        raise NotImplementedError()

    def summary(self):
        """
        Prints a string representation of the layer.
        """
        raise NotImplementedError()