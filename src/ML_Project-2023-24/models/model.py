from layers.layer import Layer
from ..metric import Metric
from ..loss import Loss
from ..optimizer import Optimizer


class Model:

    def __init__(self):
        self.optimizer = None
        self.metrics = None
        self.loss = None
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self, optimizer: Optimizer, loss: Loss, metrics: list[Metric]):
        """
        Prepares the model for fitting with an optimizer, a loss and some metrics.

        :param optimizer: the optimizer to use
        :param loss: the loss to use
        :param metrics: the list of metrics to use
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def fit(self, x, y, epochs=256, batch_size=20, verbose=False):
        """
        Fits the model to the data.

        :param x: the input data
        :param y: the target data
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
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {self.loss.evaluate(x, y)}")

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        return delta

    def predict(self, x):
        self.forward(x)

    def evaluate(self, x, y):
        y_pred = self.forward(x)
        return [metric.evaluate(y_pred, y) for metric in self.metrics]

    def train_one_step(self, x, y):
        """
        Trains the model on one batch of data.

        :param x: the input data
        :param y: the target data
        """
        y_pred = x
        for layer in self.layers:
            y_pred = layer.forward(x)

        delta = self.loss.backward(y_pred, y)

        for layer in reversed(self.layers):
            self.optimizer.update_rule(layer, delta)
            delta = layer.backward(delta)


    def summary(self):
        # ----- Implement ------
        pass
