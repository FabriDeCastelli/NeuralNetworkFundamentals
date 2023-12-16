from models.layers.layer import Layer
import numpy as np

class Optimizer:
    def __init__(self):
        pass
    
    def update_rule(self, layer:Layer, deltaW, learning_rate):
        """
        updating weights of a layer
        
        :param layer: the layer with weights to update 
        :param deltaW: the gradient of the loss with respect to the weights
        :param learning_rate: the learning rate to use
        :return: the updated weights
        """
        
        raise NotImplementedError()
    
    
    
class SGD(Optimizer):
    def __init__(self, momentum = 0, learning_rate = 0.01):
        super.__init__()
        self.momentum = momentum                #la var momentum sarebbe alpha del momentum 
        self.learning_rate = learning_rate
        pass

    def get_momentum(self):
        return self.momentum
    
    def update_rule(self, layer: Layer, delta: np.ndarray):
        """
        updating the weights of a layer
        
        :param layer: the layer with weights to update 
        :param delta: the error backpropagated from the next layer
        """   
        
        delta_w = delta * layer.get_input()
        delta_b = delta
        
        new_weights = layer.get_weights() - delta_w * self.learning_rate + self.momentum * layer.get_detla_w_old()
        new_b = layer.get_bias() - delta_b * self.learning_rate + self.momentum * layer.get_delta_b_old()
        
        layer.set_weights(new_weights)
        layer.set_bias(new_b)
        layer.set_delta_w_old(delta_w)
        layer.set_delta_b_old(delta_b)