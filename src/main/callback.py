from src.main.metric import Metric, metrics_map
from src.main.loss import Loss, loss_map


class Callback:
    """
    abstract class callback used to define callbacks objects
    """

    def __init__(self):
        pass


class EarlyStopping(Callback):
    """
    early stopping callback used to stop the training when the validation loss stops improving
    """

    def __init__(self, patience: int = 5, start_from_epoch: int = 100, delta: float = 0.001,
                 monitor: str | Metric | Loss = 'loss', restore_best_weights: bool = False, verbose: bool = False):
        """
        :param patience: the number of epochs with no improvement before stopping the training
        :param start_from_epoch: the number of epochs to wait before stopping the training
        :param delta: the minimum change in the monitored metric to qualify as an improvement
        :param monitor: the metric to monitor    
        :restore_best_weights: whether to restore model weights from the epoch with the best value of the monitored metric    
        """
        super().__init__()
        self.patience = patience
        self.star_from_epoch = start_from_epoch
        self.delta = delta
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.counter = 0
        self.val_history = []
        self.best_model = None
        self.best_iter_model = 0

    def get_counter(self):
        return self.counter

    def increment_counter(self):
        self.counter += 1

    def get_best_model(self):
        return self.best_model

    def get_restore_best_weights(self):
        return self.restore_best_weights

    def get_best_iter_model(self):
        return self.best_iter_model

    def check_stop(self):
        """
        check if the training should be stopped
        
        :return: True if the training should be stopped, False otherwise
        """

        def check(lst, delta):
            """
            Checks if all elements in the list are different by at most delta.

            :param lst: the list to check
            :param delta: the maximum allowed difference between elements
            :return: True if all elements are different by at most delta, False otherwise
            """
            n = len(lst)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(lst[i][self.monitor] - lst[j][self.monitor]) > delta:
                        return False
            return True

        if self.counter < self.star_from_epoch:
            return False

        if check(self.val_history[-self.patience:], self.delta):
            if self.verbose and self.restore_best_weights:
                print("Best model at iter: ", self.best_iter_model)
            return True
        return False

    def update_best_model(self, model, val_score: dict):
        """
        update best model
        
        :param model: the model to check if it is the best model
        :param val_score: the new validation score
        """
        if self.best_model is None:
            self.best_model = model
            self.best_iter_model = self.counter

        elif (val_score[self.monitor] < self.val_history[-1][self.monitor] and
              (self.monitor in metrics_map["minimize"] or self.monitor in loss_map["minimize"])):
            self.best_model = model
            self.best_iter_model = self.counter

        elif (val_score[self.monitor] > self.val_history[-1][self.monitor] and
              self.monitor in metrics_map["maximize"]):
            self.best_model = model
            self.best_iter_model = self.counter

    def update_val_history(self, model, val_score: dict):
        """
        update the validation history with the new validation score and store the best model

        :param model: the model to check if it is the best model
        :param val_score: the new validation score
        """
        self.val_history.append(val_score)

    def reset(self):
        """
        reset the object to the initial state
        """

        self.counter = 0
        self.val_history = []
        self.best_model = None
        self.best_iter_model = 0



callback_dict = {
    "EarlyStopping": EarlyStopping(),
}
