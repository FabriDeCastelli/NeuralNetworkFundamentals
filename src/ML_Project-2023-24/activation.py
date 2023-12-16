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
    
    

class ReLu(Activation):
    """
    ReLu activation function
    """
    
    def __init__(self):
        pass
        
    def forward(self, x):
        return np.maximum(0, x)
        
    def backward(self, x):
        return np.where(x > 0, 1, 0)

class Identity(Activation):
    """
    Identity activation function
    """
    
    def __init__(self):
        pass
    
    def forward(self, x):
        return x
    
    def backward(self, x):
        return 1
    
class Sigmoid(Activation):
    """
    Sigmoid activation function
    """
    
    def __init__(self):
        pass
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))
    
class Tanh(Activation):
    """
    Tanah activation function
    """
    
    def __init__(self):
        pass
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.tanh(x)**2