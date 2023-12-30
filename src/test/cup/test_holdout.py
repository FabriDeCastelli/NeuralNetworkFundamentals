from sklearn.model_selection import train_test_split
from src.main.evaluation.holdout_CV import holdout_CV
from src.main.evaluation.grid_search import RandomGridSearch
from src.main.utilities.utils import load_hparams, setup_experiment, log_experiment
from src.main.utilities.dataset_handler import get_cup_dataset


def print_score(mean, std):
    for key in mean.keys():
        print(key, "\t", mean[key], "+\-", std[key])


x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

hyperparameters = load_hparams("cup")
grid_search = RandomGridSearch(hyperparameters)
train_mean, train_std, val_mean, val_std, test_mean, test_std, model, histories = (
    holdout_CV(x_train, y_train, grid_search, verbose=True)
)

log_experiment(setup_experiment("cup"), model, train_mean, train_std, val_mean, val_std, test_mean, test_std, histories)

print("------ Train scores: ------ ")
print_score(train_mean, train_std)
print("------ Validation scores: ------ ")
print_score(val_mean, val_std)
print("------ Test scores: ------ ")
print_score(test_mean, test_std)

