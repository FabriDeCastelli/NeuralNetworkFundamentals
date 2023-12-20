from src.main.models.layers.layer import Layer
import numpy as np


class Optimizer:
    """
    Abstract class representing an optimizer.
    """

    def __init__(self):
        pass

    def update_parameters(self, layer: Layer):
        """
        updating weights of a layer
        
        :param layer: the layer with weights to update 
        :param delta: the gradient of the loss with respect to the weights
        :return: the updated weights
        """

        raise NotImplementedError()

    def to_string(self):
        raise NotImplementedError()
    
    def get_learning_rate(self):
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
        self.learning_rate = learning_rate
        self.momentum = momentum

    def to_string(self):
        return (
                "Stochastic Gradient Descent"
                + f"\nLearning rate: {self.learning_rate}"
                + f"\nMomentum: {self.momentum}"
        )


    def get_momentum(self):
        return self.momentum

    def get_learning_rate(self):
        return self.learning_rate

    def update_parameters(self, layer: Layer):
        """
        updating the weights of a layer
        
        :param layer: the layer with weights to update 
        :param delta: the error backpropagated from the next layer
        """

        delta = layer.get_delta()
        batch_size = delta.shape[0]

        delta_w = - self.learning_rate / batch_size * np.dot(layer.get_input().T, delta)
        delta_b = - self.learning_rate / batch_size * delta.sum(axis=0)

        new_weights = layer.get_weights() + delta_w
        new_b = layer.get_bias() + delta_b

        layer.set_weights(new_weights)
        layer.set_bias(new_b)


optimizer_dict = {
    "sgd": SGD()
}
