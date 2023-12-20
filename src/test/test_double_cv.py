
from src.main.dataset_handler import get_monk
from src.main.evaluation.grid_search import RandomGridSearch
from src.main.utils import load_hparams

x_train, y_train, x_test, y_test = get_monk(1)

hyperparameters = load_hparams("nn")
grid_search = RandomGridSearch(hyperparameters)
grid_search.run_search(x_train, y_train, verbose=True)
