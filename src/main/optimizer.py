from models.layers.layer import Layer
import numpy as np


class Optimizer:
    def __init__(self):
        pass

    def update_rule(self, layer: Layer, delta_w):
        """
        updating weights of a layer
        
        :param layer: the layer with weights to update 
        :param delta_w: the gradient of the loss with respect to the weights
        :return: the updated weights
        """

        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, momentum=0, learning_rate=0.01):
        super().__init__()
        self.momentum = momentum  # la var momentum sarebbe alpha del momentum
        self.learning_rate = learning_rate

    def get_momentum(self):
        return self.momentum

    def update_rule(self, layer: Layer, delta: np.ndarray):
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
