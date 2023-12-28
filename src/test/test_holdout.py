from src.main.evaluation.holdout_CV import holdout_CV
from src.main.evaluation.grid_search import RandomGridSearch
from src.main.utils import load_hparams
from src.main.dataset_handler import get_monk


def print_score(mean, std):
    for key in mean.keys():
        print(key, "\t", mean[key], "+\-", std[key])


x_train, y_train, x_test, y_test = get_monk(1)

hyperparameters = load_hparams("monk1")
grid_search = RandomGridSearch(hyperparameters)
train_mean, train_std, val_mean, val_std, test_mean, test_std, model = (
    holdout_CV(x_train, y_train, grid_search, verbose=False)
)

"""for i,model in enumerate(models): 
    print("------ Modello ",i+1," ------")
    print(model.summary())"""


print("------ Train scores: ------ ")
print_score(train_mean, train_std)
print("------ Validation scores: ------ ")
print_score(val_mean, val_std)
print("------ Test scores: ------ ")
print_score(test_mean, test_std)
