import json

import numpy as np
from .layers.layer import Layer
from src.main.metric import Metric, metrics_dict
from src.main.optimizer import Optimizer, optimizer_dict
from src.main.callback import Callback, callback_dict
from src.main.loss import Loss, loss_dict
from src.main.regularizer import Regularizer, regularizer_dict


class Model:

    def __init__(self):
        """
        Constructor for the model. Initializes the optimizer, the metrics and the loss to None.
        These fields will be initialized when the model is compiled.
        Layers can be added to the model with the add method.
        """
        self.optimizer = None
        self.callback = None
        self.metrics = None
        self.loss = None
        self.regularizer = None
        self.layers = []

    def get_loss(self):
        return self.loss

    def get_metrics(self):
        return self.metrics

    def add(self, layer: Layer):
        """
        Adds a layer to the model.

        :param layer: the layer to add
        """
        self.layers.append(layer)

    def compile(
            self,
            optimizer: str | Optimizer,
            loss: str | Loss,
            metrics: list[str | Metric],
            callback: str | Callback = None,
            regularizer: str | Regularizer = None,
    ):
        """
        Prepares the model for fitting with an optimizer, a loss and a list of metrics.

        :param optimizer: the optimizer to use
        :param callback: the callback to use
        :param loss: the loss to use
        :param metrics: the list of metrics to use
        :param regularizer: the regularizer to use
        """

        if isinstance(optimizer, str):
            optimizer = optimizer_dict.get(optimizer)
            if optimizer is None:
                raise ValueError("Invalid optimizer")

        if isinstance(loss, str):
            loss = loss_dict.get(loss)
            if loss is None:
                raise ValueError("Invalid loss")

        if isinstance(callback, str):
            callback = callback_dict.get(callback)
            if callback is None:
                raise ValueError("Invalid callback")

        if not isinstance(metrics, list):
            raise ValueError("Metrics must be a list")

        metrics = list(map(lambda x: metrics_dict.get(x) if isinstance(x, str) else x, metrics))

        if regularizer is not None:
            if isinstance(regularizer, str):
                regularizer = regularizer_dict.get(regularizer)
                if regularizer is None:
                    raise ValueError("Invalid regularizer")

        self.optimizer = optimizer
        self.callback = callback
        self.loss = loss
        self.metrics = metrics
        self.regularizer = regularizer

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs=256, batch_size=20, verbose=False):
        """
        Fits the model using the training data.

        :param x_train: the input data for training
        :param y_train: the target data for training
        :param x_val: the input data for validation
        :param y_val: the target data for validation
        :param epochs: the number of epochs to train the model
        :param batch_size: the size of the batch to process at each epoch
        :param verbose: whether to print the progress of the training
        """

        validation_scores = []
        training_scores = []

        for epoch in range(epochs):
            for batch in range(len(x_train) // batch_size):
                x_batch = x_train[batch * batch_size: (batch + 1) * batch_size]
                y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]

                # shuffle the data
                idx = np.random.permutation(len(x_batch))
                x_batch = x_batch[idx]
                y_batch = y_batch[idx]

                self.train_one_step(x_batch, y_batch, batch_size)

            training_scores.append(self.evaluate(x_train, y_train))

            val_score = {}
            if x_val is not None:
                val_score = self.evaluate(x_val, y_val)
                validation_scores.append(val_score)

            # early stopping logic
            if self.callback is not None and x_val is not None:

                self.callback.increment_counter()
                if self.callback.get_restore_best_weights():
                    self.callback.update_best_model(self, val_score)
                self.callback.update_val_history(val_score)

                if self.callback.check_stop():
                    print("Early Stopping Triggered at iter: ", self.callback.get_counter())
                    # self.callback.reset()
                    break

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs} \tTraining - {self.evaluate(x_train, y_train)}")
                if x_val is not None:
                    print(f"\t\tValidation - {self.evaluate(x_val, y_val)}")

        def convert_history_format(history):
            result = {}

            for key, values in history.items():
                for entry in values:
                    for subkey, subvalue in entry.items():
                        if subkey not in result:
                            result[subkey] = {}
                        if key not in result[subkey]:
                            result[subkey][key] = []
                        result[subkey][key].append(subvalue)

            return result

        history = convert_history_format({
            "training": training_scores,
            "validation": validation_scores
        })

        if self.callback and self.callback.get_restore_best_weights():
            return self.callback.get_best_model(), history

        return self, history

    def train_one_step(self, x: np.ndarray, y: np.ndarray, batch_size):
        """
        Trains the model on one batch of data.

        :param x: the input data
        :param y: the target data
        :param batch_size: the size of the batch
        """

        # Forward Pass
        y_pred = self.forward(x)

        # Compute Loss derivative to propagate back
        delta = self.loss.backward(y_pred, y)

        # Backward Pass
        self.backward(delta)

        # Update Parameters
        for layer in reversed(self.layers):
            self.optimizer.update_parameters(layer, self.regularizer, batch_size)

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

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a prediction on the input data_for_testing.

        :param x: the input data_for_testing
        :return: the prediction of the model
        """
        return self.forward(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluates the model on the input data_for_testing.

        :param x: the input data_for_testing
        :param y: the target data_for_testing
        :return: a dictionary storing the value of the metrics and the loss
        """

        y_pred = self.predict(x)

        model_score = {}

        for metric in self.metrics:
            model_score[repr(metric)] = metric.evaluate(y_pred, y)

        model_score[repr(self.loss)] = self.loss.forward(y_pred, y)

        return model_score

    def reset(self):
        """
        Reset the model to the initial state.
        """
        self.callback.reset()
        for layer in self.layers:
            layer.reset()

    def summary(self):
        """
        Prints a summary of the model.
        """
        print("-------- Model Summary --------")
        print("Model Summary:")
        print(f"Optimizer: {repr(self.optimizer)}")
        print(f"Loss: {repr(self.loss)}")
        print(f"Metrics: {list(map(lambda x: repr(x), self.metrics))}")
        print(f"Callback: {repr(self.callback)}")
        print(f"Regularizer: {repr(self.regularizer)}")
        print(" ")
        for layer in self.layers:
            layer.summary()
        print(" ")

    def to_dict(self, epochs: int, batch_size: int):
        """
        Serializes the model to a dictionary.

        :param epochs: the epochs used for training
        :param batch_size: the batch size used for training
        :return: a dictionary representing the model
        """
        model_dict = {
            "optimizer": self.optimizer.to_dict() if self.optimizer else repr(None),
            "batch_size": batch_size,
            "epochs": epochs,
            "loss": repr(self.loss),
            "metrics": [repr(metric) for metric in self.metrics],
            "callback": self.callback.to_dict() if self.callback else repr(None),
            "regularizer": self.regularizer.to_dict() if self.regularizer else repr(None),
            "layers": [layer.to_dict() for layer in self.layers]
        }
        return model_dict

    def save(self, path: str, epochs: int, batch_size: int):
        """
        Saves the model to a json file.

        :param path: the path of the file where to save the model
        :param epochs: the number of epochs used for training
        :param batch_size: the batch size used for training
        """
        model_dict = self.to_dict(epochs=epochs, batch_size=batch_size)
        with open(path, "w+") as file:
            json.dump(model_dict, file, indent=4)
