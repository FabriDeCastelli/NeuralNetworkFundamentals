import numpy as np


class Activation:
    """
    Base class for activation functions
    """

    def forward(self, x):
        """
        computes the output of the activation function
        
        :param x: the input to the activation function
        
        :return: the output of the activation function
        """
        raise NotImplementedError()

    def backward(self, x):
        """
        computes the derivative of the activation function
        
        :param x: the input to the activation function
        
        :return: the derivative of the activation function
        """
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


class ReLu(Activation):
    """
    ReLu activation function
    """

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)

    def __repr__(self):
        return self.__class__.__name__


class Identity(Activation):
    """
    Identity activation function
    """

    def forward(self, x):
        return x

    def backward(self, x):
        return 1

    def __repr__(self):
        return self.__class__.__name__


class Sigmoid(Activation):
    """
    Sigmoid activation function
    """

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))

    def __repr__(self):
        return self.__class__.__name__


class Tanh(Activation):
    """
    Tanh activation function
    """

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x) ** 2

    def __repr__(self):
        return self.__class__.__name__


activation_dict = {
    "relu": ReLu(),
    "identity": Identity(),
    "sigmoid": Sigmoid(),
    "tanh": Tanh()
}
