from sklearn.model_selection import train_test_split

from src.main.utilities.dataset_handler import get_cup_dataset
from src.main.evaluation.grid_search import RandomGridSearch
from src.main.utilities.utils import load_hparams, plot_history

x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

hyperparameters = load_hparams("cup")
grid_search = RandomGridSearch(hyperparameters)

((train_mean, train_std), (val_mean, val_std)), model, params, histories = (
    grid_search.run_search(x_train, y_train, True, combinations=50)
)
epoch, batch_size = params

for history in histories:
    plot_history(history)

print("------ Train scores: ------ ")
print(train_mean, train_std)
print("------ Validation scores: ------ ")
print(val_mean, val_std)
print("------ Test scores: ------ ")
print(model.evaluate(x_test, y_test))

print("------ Best model: ------ ")
model.summary()

print("------ Best parameters: ------ ")
print("Epoch: ", epoch)
print("Batch size: ", batch_size)
