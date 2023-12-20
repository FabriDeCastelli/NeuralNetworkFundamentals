import numpy as np


class Loss:
    """
    Base class for loss functions
    """

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

    def forward(self, y_pred, y_true):
        return np.mean(np.linalg.norm(y_pred - y_true, axis=1))

    def backward(self, y_pred, y_true):
        return y_pred - y_true / y_pred.shape[0]

    def to_string(self):
        return "Mean Euclidean Error"


class CrossEntropy(Loss):
    """
    Cross Entropy loss function
    """

    def forward(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_pred, y_true):
        return y_pred - y_true / y_pred.shape[0]

    def to_string(self):
        return "Cross Entropy"


loss_dict = {
    "mean_squared_error": MSE(),
    "mean_euclidean_error": MEE(),
    "cross_entropy": CrossEntropy()
}
