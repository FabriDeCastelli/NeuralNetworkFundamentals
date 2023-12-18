from src.main.models.layers.layer import Layer
import numpy as np


class Optimizer:
    """
    Abstract class representing an optimizer.
    """
    def __init__(self):
        pass

    def update_parameters(self, layer: Layer, delta: np.ndarray):
        """
        updating weights of a layer
        
        :param layer: the layer with weights to update 
        :param delta: the gradient of the loss with respect to the weights
        :return: the updated weights
        """

        raise NotImplementedError()


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer.
    """
    def __init__(self, learning_rate=0.01, momentum=0):
        """
        Constructor for the SGD optimizer.

        :param momentum: the momentum term
        :param learning_rate: the learning rate
        """
        super().__init__()
        self.momentum = momentum
        self.learning_rate = learning_rate

    def get_momentum(self):
        return self.momentum

    def get_learning_rate(self):
        return self.learning_rate

    def update_parameters(self, layer: Layer, delta: np.ndarray):
        """
        updating the weights of a layer
        
        :param layer: the layer with weights to update 
        :param delta: the error backpropagated from the next layer
        """

        delta_w = np.dot(layer.get_input().T, delta)
        delta_b = delta.sum(axis=0)

        """
        print("--------------------")
        print("delta_w", delta_w)
        print("delta_b", delta_b)
        """

        delta_w = - self.learning_rate * delta_w + self.momentum * layer.get_delta_w_old()
        delta_b = - self.learning_rate * delta_b + self.momentum * layer.get_delta_b_old()

        new_weights = layer.get_weights() + delta_w
        new_b = layer.get_bias() + delta_b

        layer.set_weights(new_weights)
        layer.set_bias(new_b)
        layer.set_delta_w_old(delta_w)
        layer.set_delta_b_old(delta_b)


optimizer_dict = {
    "SGD": SGD()
}