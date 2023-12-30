from src.main.evaluation.double_kfold_cross_validation import double_Kfold_CV
from src.main.evaluation.grid_search import RandomGridSearch
from src.main.utilities.utils import load_hparams
from src.main.utilities.dataset_handler import get_monk


def print_score(mean, std):
    for key in mean.keys():
        print(key, "\t", mean[key], "+\-", std[key])


x_train, y_train, x_test, y_test = get_monk(1)

hyperparameters = load_hparams("monk1")
grid_search = RandomGridSearch(hyperparameters)
train_mean, train_std, val_mean, val_std, test_mean, test_std, models = (
    double_Kfold_CV(x_train, y_train, grid_search, 5, verbose=True)
)

for model in models:
    print(model.summary())


print("------ Train scores: ------ ")
print_score(train_mean, train_std)
print("------ Validation scores: ------ ")
print_score(val_mean, val_std)
print("------ Test scores: ------ ")
print_score(test_mean, test_std)
