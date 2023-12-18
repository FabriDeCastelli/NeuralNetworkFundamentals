import numpy as np


class Metric:
    """
    Base class for evaluation metrics
    """

    def __init__(self):
        pass

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

    def __init__(self):
        super().__init__()

    def evaluate(self, y_pred, y_true):
        return np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    def to_string(self):
        return "RMSE"


class Accuracy(Metric):
    """
    Accuracy evaluation metric
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, y_pred, y_true):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
    
    def to_string(self):
        return "Accuracy"


metrics_dict = {
    "RMSE": RootMeanSquaredError(),
    "Accuracy": Accuracy()
}
