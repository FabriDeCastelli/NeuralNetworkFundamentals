import numpy as np
from .layers.layer import Layer
from src.main.metric import Metric
from src.main.optimizer import Optimizer
from src.main.loss import Loss


class Model:

    def __init__(self):
        """
        Constructor for the model. Initializes the optimizer, the metrics and the loss to None.
        These fields will be initialized when the model is compiled.
        Layers can be added to the model with the add method.
        """
        self.optimizer = None
        self.metrics = None
        self.loss = None
        self.layers = []

    def add(self, layer: Layer):
        """
        Adds a layer to the model.

        :param layer: the layer to add
        """
        self.layers.append(layer)

    def compile(self, optimizer: Optimizer, loss: Loss, metrics: list[Metric]):
        """
        Prepares the model for fitting with an optimizer, a loss and a list of metrics.

        :param optimizer: the optimizer to use
        :param loss: the loss to use
        :param metrics: the list of metrics to use
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=256, batch_size=20, verbose=False):
        """
        Fits the model to the data_for_testing.

        :param x: the input data_for_testing
        :param y: the target data_for_testing
        :param epochs: the number of epochs to train the model
        :param batch_size: the size of the batch to process at each epoch
        :param verbose: whether to print the progress of the training
        """
        for epoch in range(epochs):
            for batch in range(len(x) // batch_size):
                x_batch = x[batch * batch_size: (batch + 1) * batch_size]
                y_batch = y[batch * batch_size: (batch + 1) * batch_size]
                self.train_one_step(x_batch, y_batch)

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {self.evaluate(x, y)}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the model, by calling the forward method of each layer.

        :param x: the input data_for_testing
        :return: the prediction of the model
        """
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Performs a backward pass of the model, by calling the backward method of each layer.

        :param delta: the error to propagate back
        :return: the error to propagate to the previous layer
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        return delta

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a prediction on the input data_for_testing.

        :param x: the input data_for_testing
        :return: the prediction of the model
        """
        return self.forward(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluates the model on the input data_for_testing.

        :param x: the input data_for_testing
        :param y: the target data_for_testing
        :return: the value of the loss and the metrics
        """

        y_pred = self.predict(x)
        for a, b in zip(y_pred, y):
            print(a, b)
        return [metric.evaluate(y_pred, y) for metric in self.metrics]

    def train_one_step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Trains the model on one batch of data_for_testing.

        :param x: the input data_for_testing
        :param y: the target data_for_testing
        """
        y_pred = x
        for layer in self.layers:
            y_pred = layer.forward(y_pred)

        delta = self.loss.backward(y_pred, y)

        for layer in reversed(self.layers):
            self.optimizer.update_parameters(layer, delta)
            delta = layer.backward(delta)

    def summary(self):
        """
        Prints a summary of the model.
        """
        print("Model summary:")
        print(f"Optimizer: {self.optimizer}")
        print(f"Loss: {self.loss}")
        print(f"Metrics: {self.metrics}")
        print("Layers:")
        for layer in self.layers:
            layer.summary()
