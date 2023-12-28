import numpy as np


class Metric:
    """
    Base class for evaluation metrics
    """

    def to_string(self):
        raise NotImplementedError()

    def evaluate(self, y_pred, y_true):
        """
        computes the error of the prediction.
        
        :param y_pred: the predicted values
        :param y_true: the true values
        
        :return: the error of the prediction
        """
        raise NotImplementedError()


class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error evaluation metric
    """

    def evaluate(self, y_pred, y_true):
        return np.sqrt(np.mean(np.sum(np.square(np.subtract(y_pred, y_true)), axis=1), axis=0))

    def to_string(self):
        return "root_mean_squared_error"


class Accuracy(Metric):
    """
    Accuracy evaluation metric
    """

    def evaluate(self, y_pred, y_true):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    def to_string(self):
        return "accuracy"


class BinaryAccuracy(Metric):
    """
    Binary Accuracy evaluation metric
    """

    def evaluate(self, y_pred, y_true, threshold=0.7):
        y_pred = np.where(y_pred > threshold, 1, 0)
        return np.mean(y_pred == y_true)

    def to_string(self):
        return "binary_accuracy"


metrics_dict = {
    "root_mean_squared_error": RootMeanSquaredError(),
    "accuracy": Accuracy(),
    "binary_accuracy": BinaryAccuracy(),
}

# hash map to know which metric to maximize and which to minimize
metrics_map = {
    "maximize": ["accuracy", "binary_accuracy"],
    "minimize": ["root_mean_squared_error"],
}
