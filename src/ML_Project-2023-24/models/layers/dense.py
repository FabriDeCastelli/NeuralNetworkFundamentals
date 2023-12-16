import numpy as np
from layer import Layer
from initializer import Initializer
from activation import Activation


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
        rangeMin: float,
        rangeMax: float,
        activation: Activation,
    ):
        
        """
        :param input_size: size of the input to the layer
        :param output_size: size of the output of the layer
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = initializer.weight_init((input_size, output_size),rangeMin, rangeMax)
        self.bias = initializer.bias_init((output_size))
        self.activation = activation
        # input before activation
        self.net = None             
        self.input = None
        self.delta_w_old = np.zeros((input_size, output_size))
        self.delta_b_old = np.zeros((output_size))
        
        
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def get_input(self):
        return self.input
    
    def get_detla_w_old(self):
        return self.deltaW_old
    
    def get_delta_b_old(self):
        return self.delta_b_old
    
    def set_delta_w_old(self, delta_w):
        self.deltaW_old = delta_w
        
    def set_delta_b_old(self, delta_b):
        self.delta_b_old = delta_b
        
    def set_weights(self, new_weights):
        self.weights = new_weights
    
    def set_bias(self, new_bias):
        self.bias = new_bias
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer.

        :param x: input to the layer
        :return: output of the layer
        """
        self.input = x
        self.net =  self.activation.forward(np.dot(self.weights.T, x) + self.bias)
        return self.activation.forward(self.net)

    def backward(self, delta: np.ndarray):
        """
        Performs a backward pass of the layer.
        
        :param delta: error propagated by next layer
        :return: error to prop to the previous layer
        """
        
        return np.dot(delta, self.weights.T) * self.activation.backward(self.net)

    