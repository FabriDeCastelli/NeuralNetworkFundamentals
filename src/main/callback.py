from src.main.metric import Metric, metrics_map
from src.main.loss import Loss, loss_map


class Callback:
    """
    abstract class callback used to define callbacks objects
    """

    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class EarlyStopping(Callback):
    """
    early stopping callback used to stop the training when the validation loss stops improving
    """

    def __init__(self,
                 patience: int = 5,
                 start_from_epoch: int = 100,
                 delta: float = 0.0001,
                 monitor: str = None,
                 restore_best_weights: bool = False,
                 verbose: bool = True
                 ):
        """
        :param patience: the number of epochs with no improvement before stopping the training
        :param start_from_epoch: the number of epochs to wait before stopping the training
        :param delta: the minimum change in the monitored metric to qualify as an improvement
        :param monitor: the metric to monitor    
        :param restore_best_weights: whether to restore model weights from the epoch with the best value of the monitored metric
        """
        super().__init__()

        self.patience = patience
        self.start_from_epoch = start_from_epoch
        self.delta = delta
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.compare_op = lambda a, b: a - b <= self.delta
        if self.monitor in metrics_map['maximize']:
            self.compare_op = lambda a, b: a - b >= self.delta

        self.counter = 0
        self.best_model = None
        self.best_iter_model = 0
        self.best_score = {}
        self.not_improved_count = 0

    def __call__(self, model, val_score: dict):
        """
        Implementation of Early Stopping logic

        :param model: the model to check if it is the best model
        :param val_score: the new validation score
        """

        if self.monitor is None:
            raise ValueError("The monitor metric must be defined")

        self.counter += 1

        if self.counter < self.start_from_epoch:
            return

        if self.counter == self.start_from_epoch:
            self.best_score = val_score
            self.update_best_model(model)
            return

        if self.compare_op(val_score[self.monitor], self.best_score[self.monitor]):
            self.best_score = val_score
            self.not_improved_count = 0
            self.update_best_model(model)
        else:
            self.not_improved_count += 1

        if self.not_improved_count >= self.patience:
            if self.verbose:
                print("Early stopping at epoch {}".format(self.counter))
            raise StopIteration()

    def get_best_model(self):
        return self.best_model

    def get_restore_best_weights(self):
        return self.restore_best_weights

    def update_best_model(self, model):
        """
        update best model
        
        :param model: the model to check if it is the best model
        """

        self.best_model = model
        self.best_iter_model = self.counter

    def reset(self):
        """
        Resets the object to the initial state.
        """

        self.counter = 0
        self.best_score = {}
        self.best_model = None
        self.best_iter_model = 0
        self.not_improved_count = 0

    def __repr__(self):
        return "Early Stopping"

    def to_dict(self):
        return {
            "Patience": self.patience,
            "Start From Epoch": self.start_from_epoch,
            "Delta": self.delta,
            "Monitor": self.monitor,
            "Restore Best Weights": self.restore_best_weights
        }


callback_dict = {
    "EarlyStopping": EarlyStopping(),
}
