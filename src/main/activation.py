import numpy as np


class Activation:
    """
    Base class for activation functions
    """

    def __init__(self):
        pass

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

    def to_string(self):
        raise NotImplementedError()



class ReLu(Activation):
    """
    ReLu activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)

    def to_string(self):
        return "ReLu"


class Identity(Activation):
    """
    Identity activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def backward(self, x):
        return 1

    def to_string(self):
        return "Identity"


class Sigmoid(Activation):
    """
    Sigmoid activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))

    def to_string(self):
        return "sigmoid"


class Tanh(Activation):
    """
    Tanh activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x) ** 2

    def to_string(self):
        return "Tanh"


activation_dict = {
    "relu": ReLu(),
    "identity": Identity(),
    "sigmoid": Sigmoid(),
    "tanh": Tanh()
}
