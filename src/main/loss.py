import numpy as np

from src.main.metric import Metric


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

    def __repr__(self):
        raise NotImplementedError()


class MSE(Loss, Metric):
    """
    Mean Squared Error loss function
    """

    def forward(self, y_pred, y_true):
        return np.mean(np.sum(np.square(np.subtract(y_pred, y_true)), axis=1), axis=0)

    def backward(self, y_pred, y_true):
        return np.subtract(y_pred, y_true)

    def __repr__(self):
        return "Mean Squared Error"


class MEE(Loss, Metric):
    """
    Mean Euclidean Error loss function
    """

    def forward(self, y_pred, y_true):
        return np.mean(np.linalg.norm(np.subtract(y_true, y_pred), ord=2, axis=1), axis=0)

    def backward(self, y_pred, y_true):
        return np.subtract(y_pred, y_true) / self.forward(y_pred, y_true)

    def evaluate(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def __repr__(self):
        return "Mean Euclidean Error"


class CrossEntropy(Loss):
    """
    Cross Entropy loss function
    """

    def forward(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_pred, y_true):
        return y_pred - y_true / y_pred.shape[0]

    def __repr__(self):
        return "Cross Entropy"


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy loss function
    """

    def forward(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))

    def backward(self, y_pred, y_true):
        return y_pred - y_true / y_pred.shape[0]

    def __repr__(self):
        return "Binary Cross Entropy"


loss_dict = {
    "mean_squared_error": MSE(),
    "mean_euclidean_error": MEE(),
    "cross_entropy": CrossEntropy(),
    "binary_cross_entropy": BinaryCrossEntropy()
}

# hash map to know which metric to maximize and which to minimize (always minimize the loss)
loss_map = {
    "minimize": ["loss"]
}