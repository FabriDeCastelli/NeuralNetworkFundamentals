from src.main.models.layers.layer import Layer
import numpy as np

from src.main.regularizer import Regularizer


class Optimizer:
    """
    Abstract class representing an optimizer.
    """

    def __init__(self):
        pass

    def update_parameters(self, layer: Layer, regularizer: Regularizer = None):
        """
        updating weights of a layer
        
        :param layer: the layer with weights to update 
        :param regularizer: the regularizer to use
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

    def update_parameters(self, layer: Layer, regularizer: Regularizer = None):
        """
        updating the weights of a layer
        
        :param layer: the layer with weights to update
        :param regularizer: the regularizer to use
        """

        delta = layer.get_delta()
        batch_size = delta.shape[0]

        dW = self.learning_rate * np.dot(layer.get_input().T, delta) / batch_size
        db = self.learning_rate * delta.sum(axis=0) / batch_size

        if regularizer is not None:
            dW += regularizer.backward(layer.get_weights())

        if self.momentum > 0:
            dW += self.momentum * layer.get_dW()
            db += self.momentum * layer.get_db()
            layer.set_dW(dW)
            layer.set_db(db)

        new_weights = layer.get_weights() - dW
        new_b = layer.get_bias() - db

        layer.set_weights(new_weights)
        layer.set_bias(new_b)


optimizer_dict = {
    "sgd": SGD()
}
