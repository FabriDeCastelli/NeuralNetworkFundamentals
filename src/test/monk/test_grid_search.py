from src.main.dataset_handler import get_monk
from src.main.evaluation.grid_search import RandomGridSearch
from src.main.utils import load_hparams, plot_history

x_train, y_train, x_test, y_test = get_monk(1)

hyperparameters = load_hparams("monk1")
grid_search = RandomGridSearch(hyperparameters)


((train_mean, train_std), (val_mean, val_std)), model, params, histories = (
    grid_search.run_search(x_train, y_train, False, combinations=50))
epoch, batch_size = params

for i,history in enumerate(histories):
    print("History for fold", i+1, ":")
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
