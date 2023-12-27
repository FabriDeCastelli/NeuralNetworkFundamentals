import numpy as np


class Callback:
    """
    abstract class callback used to define callbacks objects
    """
    
    
    def __init__():
        pass
        

class EarlyStopping(Callback):
    """
    early stopping callback used to stop the training when the validation loss stops improving
    """
    
    def __init__(
        self, 
        start_from_epoch: int = 150, 
        number_of_iter: int = 5,
        delta: float = 0.001,
        monitor: str = 'val_loss',
    ):
        """
        :param patience: the number of epochs to wait before stopping the training
        :param delta: the minimum change in the monitored metric to qualify as an improvement
        :param stopping_metric: the metric to monitor        
        """
        self.counter = 0
        self.val_history = []
        self.patience = start_from_epoch
        self.number_of_iter = number_of_iter
        self.delta = delta
        self.monitor = monitor
          
    def get_counter(self):
        return self.counter
    
    def increment_counter(self):
        self.counter += 1      
                
    def check_stop(self):
        
        """
        def check(list, delta):
            t = []
            for el in list:
                t.append(el[self.stopping_metric])  
            return max(t) - min(t) <= delta
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
                        return False  # Difference exceeds delta, elements are not within the threshold
            return True  # All elements are within the threshold delta
            
        if(self.counter < self.patience):
            return False
    
        if(check(self.val_history[-self.number_of_iter:], self.delta)):
            return True
        return False     
    
    def update_val_history(self, val_score: dict):
        self.val_history.append(val_score)
        
    def reset(self):
        self.counter = 0
        self.val_history = []
    
    
    
callback_dict = {
    "EarlyStopping": EarlyStopping(),
}



