from src.main.evaluation.holdout_CV import holdout_CV
from src.main.evaluation.grid_search import GridSearch
from src.main.utilities.utils import load_hparams, setup_experiment, log_experiment, predictions_to_csv
from src.main.utilities.dataset_handler import get_cup_dataset


def print_score(mean, std):
    for key in mean.keys():
        print(key, "\t", mean[key], "+\-", std[key])


x_train, y_train, x_test = get_cup_dataset()

hyperparameters = load_hparams("cup")
grid_search = GridSearch(hyperparameters)
train_mean, train_std, val_mean, val_std, test_mean, test_std, model, (epochs, batch_size), histories = (
    holdout_CV(x_train, y_train, grid_search, verbose=False)
)

log_experiment(
    setup_experiment("cup_submission"),
    model, epochs, batch_size,
    train_mean, train_std, val_mean, val_std, test_mean, test_std, histories
)

predictions = model.predict(x_test)
predictions_to_csv(predictions)

print("------ Train scores: ------ ")
print_score(train_mean, train_std)
print("------ Validation scores: ------ ")
print_score(val_mean, val_std)
print("------ Test scores: ------ ")
print_score(test_mean, test_std)
