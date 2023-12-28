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
        return np.mean(np.sum(np.square(np.subtract(y_pred, y_true)), axis=1), axis=0)

    def backward(self, y_pred, y_true):
        return 2 * np.subtract(y_pred, y_true)

    def to_string(self):
        return "mean_squared_error"


class MEE(Loss):
    """
    Mean Euclidean Error loss function
    """

    def forward(self, y_pred, y_true):
        return np.sqrt(np.mean(np.sum(np.square(np.subtract(y_pred, y_true)), axis=1), axis=0))

    def backward(self, y_pred, y_true):
        return np.subtract(y_pred, y_true) / np.linalg.norm(y_pred - y_true, ord=2)

    def to_string(self):
        return "mean_euclidean_error_loss"


class CrossEntropy(Loss):
    """
    Cross Entropy loss function
    """

    def forward(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_pred, y_true):
        return y_pred - y_true / y_pred.shape[0]

    def to_string(self):
        return "cross_entropy"


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy loss function
    """

    def forward(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))

    def backward(self, y_pred, y_true):
        return y_pred - y_true / y_pred.shape[0]

    def to_string(self):
        return "binary_cross_entropy"


loss_dict = {
    "mean_squared_error": MSE(),
    "mean_euclidean_error_loss": MEE(),
    "cross_entropy": CrossEntropy(),
    "binary_cross_entropy": BinaryCrossEntropy()
}

# hash map to know which metric to maximize and which to minimize (always minimize the loss)
loss_map = {
    "minimize": ["loss"]
}