import numpy as np
from layer import Layer


class Dense(Layer):
    """
    A fully connected layer.
    Defined with an input size and an output size.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        :param input_size: size of the input to the layer
        :param output_size: size of the output of the layer
        """
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer.

        :param x: input to the layer
        :return: output of the layer
        """
        # ----- Implement ------

        pass

    def backward(self, error: np.ndarray):
        # ----- Implement ------

        pass
