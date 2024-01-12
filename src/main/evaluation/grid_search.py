import numpy as np
from joblib import Parallel, delayed
from src.main.utilities.utils import load_hparams, create_model, plot_history
import itertools
from .kfold_cross_validation import Kfold_CV
from sklearn.model_selection import train_test_split


class GridSearch:
    """
    Class used to define GridSearch objects
    """

    def __init__(self, parameters):
        """
        Constructor of the class

        :param parameters: the dictionary of parameters to be used in the grid search
        """
        for param in parameters:
            setattr(self, param, parameters[param])

    def get_parameters_combination(self):
        """
        Function that returns the parameters combination

        :return: the parameters combination
        """
        keys, values = zip(*self.__dict__.items())
        # remove the combinations from keys and values
        keys = keys[:-1]
        values = values[:-1]
        result = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return list(filter(lambda x: len(x['units']) - 1 == len(x['activations']), result))

    def run_search(self, x, y, kfold=True, verbose=False):
        """
        Performs a grid search over the parameters stored in the instance.
        At each parameter configuration a K Fold CV is performed and the
        average of the scores is stored.

        :param x: the input data
        :param y: the target data
        :param kfold: whether to use kfold or holdout
        :param verbose: whether to print the progress of the training
        :return: the best model found
        """
        parameters_combination = self.get_parameters_combination()
        return GridSearch.search(x, y, parameters_combination, verbose, kfold=kfold)

    @staticmethod
    def search(x, y, parameters_combination, verbose=False, kfold=True):
        """
        Performs a grid search over the parameters stored in the instance.
        At each parameter configuration a K Fold CV is performed and the
        average of the scores is stored.

        :param x: the input data
        :param y: the target data
        :param parameters_combination: the parameters combination to be used in the grid search
        :param verbose: whether to print the progress of the training
        :param kfold: whether to use kfold or holdout
        :return: the best score, model and params for fitting found
        """
        assert parameters_combination

        def run(parameters, verbose, kfold=True):
            model = create_model(**parameters)
            batch_size = parameters['batch_size']
            epochs = parameters['epochs']
            if kfold:
                return Kfold_CV(x, y, model, 7, epochs, batch_size, verbose), model, (epochs, batch_size)
            else:
                x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)
                model, history = model.fit(x_train, y_train, x_val, y_val, epochs, batch_size, verbose)

                train_score = model.evaluate(x_train, y_train)
                val_score = model.evaluate(x_val, y_val)
                train_std = {}
                val_std = {}

                for key in train_score.keys():
                    train_std[key] = 0.0
                    val_std[key] = 0.0

                return (((train_score, train_std), (val_score, val_std)), [history]), model, (epochs, batch_size)

        results = Parallel(n_jobs=-1)(
            delayed(run)(parameters, verbose, kfold) for parameters in parameters_combination
        )

        best_model = None
        best_params = None
        best_scores = None
        best_history = None
        best_val_loss = np.inf
        top5 = []

        for i, (res, model, parameters) in enumerate(results):
            result, histories = res
            mean_val = result[1][0][repr(model.get_loss())]
            top5.append((result, model, parameters))
            if mean_val < best_val_loss:
                best_val_loss = mean_val
                best_scores = result
                best_model = model
                best_params = parameters
                best_history = histories

        # Sort by increasing validation loss
        top5 = sorted(top5, key=lambda e: e[0][1][0][repr(model.get_loss())])[:5]
        return best_scores, best_model, best_params, best_history, top5


class RandomGridSearch(GridSearch):
    """
    Class that implements a random grid search
    """

    def __init__(self, params, combinations=10):
        """
        Constructor of the class

        :param params: the dictionary of parameters to be used in the grid search
        """
        super().__init__(params)
        self.combinations = combinations

    def get_parameters_combination(self):
        """
        Function that returns the random parameters combination

        :return: the parameters combination
        """
        all_params_combination = super().get_parameters_combination()
        total_combinations = len(all_params_combination)
        return [
            all_params_combination[i] for i in np.random.choice(total_combinations, self.combinations)
        ]

    def run_search(self, x, y, verbose=False, kfold=True):
        parameters_combination = self.get_parameters_combination()
        return GridSearch.search(x, y, parameters_combination, verbose, kfold)
