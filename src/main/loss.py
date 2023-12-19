import numpy as np


class Loss:
    """
    Base class for loss functions
    """

    def __init__(self):
        pass

    def to_string(self):
        raise NotImplementedError()

    def forward(self, y_pred, y_true):
        """
        computes the error of the prediction
        
        :param y_pred: the predicted values
        :param y_true: the true values
        
        :return: the error of the prediction
        """

        raise NotImplementedError()

    def backward(self, y_pred, y_true):
        """
        computes the derivative of the loss function
        
        :param y_pred: the predicted values
        :param y_true: the true values
        
        :return: the derivative of the loss function
        """

        raise NotImplementedError()


class MSE(Loss):
    """
    Mean Squared Error loss function
    """

    def __init__(self):
        super().__init__()

    def to_string(self):
        return "Mean Squared Error"

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size
    
    def to_string(self):
        return "MSE"


class MEE(Loss):
    """
    Mean Euclidean Error loss function
    """

    def __init__(self):
        super().__init__()

    def to_string(self):
        return "Mean Euclidean Error"

    def forward(self, y_pred, y_true):
        return np.mean(np.linalg.norm(y_pred - y_true, axis=1))

    def backward(self, y_pred, y_true):
        return y_pred - y_true / y_pred.shape[0]
    
    def to_string(self):
        return "MEE"


loss_dict = {
    "MSE": MSE(),
    "MEE": MEE()
}