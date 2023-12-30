import numpy as np
from .layer import Layer
from src.main.initializer import initializer_dict, Initializer
from src.main.activation import activation_dict, Activation


class Dense(Layer):
    """
    A fully connected layer.
    Defined with an input size and an output size.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            weight_initializer: str | Initializer = 'glorot_uniform',
            bias_initializer: str | Initializer = 'zeros',
            activation: str | Activation = 'identity',
    ):
        """
        Constructor for the Dense layer.

        :param input_size: size of the input to the layer
        :param output_size: size of the output of the layer
        :param weight_initializer: initializer for the weights
        :param bias_initializer: initializer for the bias
        :param activation: activation function to use
        """

        if input_size <= 0 or output_size <= 0:
            raise ValueError("Invalid input/output size of the layer")

        self.input_size = input_size
        self.output_size = output_size

        if isinstance(weight_initializer, str):
            weight_initializer = initializer_dict.get(weight_initializer)
            if weight_initializer is None:
                raise ValueError("Invalid weight initializer")

        if isinstance(bias_initializer, str):
            bias_initializer = initializer_dict.get(bias_initializer)
            if bias_initializer is None:
                raise ValueError("Invalid bias initializer")

        self.weights_initializer = weight_initializer
        self.weights = weight_initializer((input_size, output_size))

        self.bias_initializer = bias_initializer
        self.bias = bias_initializer((output_size,))

        if isinstance(activation, str):
            activation = activation_dict.get(activation)
            if activation is None:
                raise ValueError("Invalid activation function")

        self.activation = activation

        # Backpropagation variables
        self.input = None
        self.output = None
        self.delta = None
        self.dW = np.zeros((input_size, output_size))
        self.db = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer.

        :param x: input to the layer
        :return: output of the layer
        """
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.activation.forward(self.output)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Performs a backward pass of the layer.
        
        :param delta: error propagated by next layer
        :return: error to propagate to the previous layer
        """

        self.delta = delta * self.activation.backward(self.output)
        return self.delta.dot(self.weights.T)

    def summary(self):
        """
        Prints a summary of the layer
        """
        print("-------- Dense Layer --------")
        print("Input size: ", self.input_size)
        print("Output size: ", self.output_size)
        print("Weights initializer: ", repr(self.weights_initializer))
        print("Bias initializer: ", repr(self.bias_initializer))
        print("Activation: ", repr(self.activation))
        print("Parameters: ", self.weights.shape[0] * self.weights.shape[1] + self.bias.shape[0])
        
    def reset(self):
        """
        Reset the layer
        """
        self.weights = self.weights_initializer((self.input_size, self.output_size))
        self.bias = self.bias_initializer((self.output_size,))
        self.input = None
        self.output = None
        self.delta = None
        self.dW = np.zeros((self.input_size, self.output_size))
        self.db = np.zeros(self.output_size)

    # getters and setters
    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_input(self):
        return self.input
    
    def get_input_size(self):
        return self.input_size
    
    def get_output_size(self):
        return self.output_size 

    def get_delta(self):
        return self.delta

    def get_dW(self):
        return self.dW

    def get_db(self):
        return self.db

    def set_dW(self, dW):
        self.dW = dW

    def set_db(self, db):
        self.db = db

    def set_weights(self, new_weights):
        self.weights = new_weights

    def set_bias(self, new_bias):
        self.bias = new_bias
        
    def get_weights_initializer(self):
        return self.weights_initializer

