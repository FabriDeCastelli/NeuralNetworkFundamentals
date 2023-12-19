import numpy as np
from .layer import Layer
from src.main.activation import Activation
from src.main.initializer import Initializer


class Dense(Layer):
    """
    A fully connected layer.
    Defined with an input size and an output size.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            initializer: Initializer,
            range: (float, float),
            activation: Activation,
    ):
        """
        Constructor for the Dense layer.

        :param input_size: size of the input to the layer
        :param output_size: size of the output of the layer
        :param initializer: initializer for the weights and biases
        :param range: range for the weights and biases
        :param activation: activation function to use
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = initializer.weight_init((input_size, output_size), range)
        self.bias = initializer.bias_init((output_size,))
        self.activation = activation
        # input before activation
        self.net = None
        self.input = None
        self.delta_w_old = np.zeros((input_size, output_size))
        self.delta_b_old = np.zeros((output_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer.

        :param x: input to the layer
        :return: output of the layer
        """
        self.input = x
        self.net = np.dot(x, self.weights) + self.bias
        return self.activation.forward(self.net)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Performs a backward pass of the layer.
        
        :param delta: error propagated by next layer
        :return: error to prop to the previous layer
        """

        """
        print("delta", delta.shape)
        print("weights", self.weights.shape)
        print("net", self.net.shape)
        """
        return np.dot(delta * self.activation.backward(self.net), self.weights.T)

    def summary(self):
        """
        Prints a summary of the layer
        """
        print("-------- Dense Layer --------")
        print("Input size: ", self.input_size)
        print("Output size: ", self.output_size)
        print("Activation: ", self.activation.to_string())
        print("Parameters: ", self.weights.shape[0] * self.weights.shape[1] + self.bias.shape[0])

    # getters and setters
    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_input(self):
        return self.input

    def get_delta_w_old(self):
        return self.delta_w_old

    def get_delta_b_old(self):
        return self.delta_b_old

    def set_delta_w_old(self, delta_w):
        self.delta_w_old = delta_w

    def set_delta_b_old(self, delta_b):
        self.delta_b_old = delta_b

    def set_weights(self, new_weights):
        self.weights = new_weights

    def set_bias(self, new_bias):
        self.bias = new_bias
