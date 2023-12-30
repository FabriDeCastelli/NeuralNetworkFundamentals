import numpy as np
from joblib import Parallel, delayed
from src.main.utilities.utils import load_hparams, create_model
import itertools
from .kfold_cross_validation import Kfold_CV


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
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def run_search(self, x, y, verbose=False, combinations=-1):
        """
        Performs a grid search over the parameters stored in the instance.
        At each parameter configuration a K Fold CV is performed and the
        average of the scores is stored.

        :param x: the input data
        :param y: the target data
        :param verbose: whether to print the progress of the training
        :param combinations: not used
        :return: the best model found
        """
        parameters_combination = self.get_parameters_combination()
        return GridSearch.search(x, y, parameters_combination, verbose)

    @staticmethod
    def search(x, y, parameters_combination, verbose=False):
        """
        Performs a grid search over the parameters stored in the instance.
        At each parameter configuration a K Fold CV is performed and the
        average of the scores is stored.

        :param x: the input data
        :param y: the target data
        :param parameters_combination: the parameters combination to be used in the grid search
        :param verbose: whether to print the progress of the training
        :return: the best score, model and params for fitting found
        """
        assert parameters_combination

        def run(parameters, verbose):
            # print(f"Running with parameters: {parameters}")
            model = create_model(**parameters)
            batch_size = parameters['batch_size']
            epochs = parameters['epochs']
            return Kfold_CV(x, y, model, 7, epochs, batch_size, verbose), (epochs, batch_size)

        results = Parallel(n_jobs=7)(
            delayed(run)(parameters, verbose) for parameters in parameters_combination
        )

        best_model = None
        best_params = None
        best_scores = None
        best_history = None
        best_val_loss = np.inf

        for res, parameters in results:
            result, model, histories = res
            # print(result[1][0])
            mean_val = result[1][0][repr(model.get_loss())]
            if mean_val < best_val_loss:
                best_val_loss = mean_val
                best_scores = result
                best_model = model
                best_params = parameters
                best_history = histories

        return best_scores, best_model, best_params, best_history


class NestedGridSearch(GridSearch):
    """
    Class that implements a nested grid search.
    """

    def __init__(self, params):
        """
        Constructor of the class

        :param params: the dictionary of parameters to be used in the grid search
        """
        super().__init__(params)

    def run_search(self, **kwargs):
        """
        function that do nested grid search by calling two times
        evaluate_candidates over two different distribution of parameters
        """
        pass


class RandomGridSearch(GridSearch):
    """
    Class that implements a random grid search
    """

    def __init__(self, params):
        """
        Constructor of the class

        :param params: the dictionary of parameters to be used in the grid search
        """
        super().__init__(params)

    def get_parameters_combination(self, combinations=1):
        """
        Function that returns the random parameters combination

        :param combinations: the number of combinations to be returned
        :return: the parameters combination
        """
        all_params_combination = super().get_parameters_combination()
        total_combinations = len(all_params_combination)
        return [
            all_params_combination[i] for i in np.random.choice(total_combinations, combinations)
        ]

    def run_search(self, x, y, verbose=False, combinations=1):
        parameters_combination = self.get_parameters_combination(combinations=combinations)
        return super().search(x, y, parameters_combination, verbose)


if __name__ == '__main__':
    params = load_hparams("monk1")
    grid_search = RandomGridSearch(params)
    grid_search.run_search(None, None, None)
    # print(grid_search.get_parameters_combination(2))
