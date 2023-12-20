
from src.main.dataset_handler import get_monk
from src.main.evaluation.grid_search import RandomGridSearch
from src.main.utils import load_hparams

x_train, y_train, x_test, y_test = get_monk(1)

hyperparameters = load_hparams("nn")
grid_search = RandomGridSearch(hyperparameters)
grid_search.run_search(x_train, y_train, verbose=True)


"""
---- TESTING KFOLD CV

from src.main.dataset_handler import get_monk
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
from src.main.evaluation.kfold_cross_validation import Kfold_CV
x_train, y_train, x_test, y_test = get_monk(1)

model = Model()
model.add(Dense(17, 4, activation="relu"))
model.add(Dense(4, 1, activation="sigmoid"))

optimizer = SGD(learning_rate=0.9, momentum=0.1)

model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["binary_accuracy"])

#model.fit(x_train, y_train, epochs=1500, batch_size=12, verbose=True)

train_score, val_score = Kfold_CV(x_train, y_train, model, 10, 1501, 12, verbose=True)

print(model.evaluate(x_test, y_test))


---- TESTING DOUBLE KFOLD CV


from src.main.dataset_handler import get_monk
from src.main.evaluation.grid_search import RandomGridSearch
from src.main.utils import load_hparams
from src.main.evaluation.double_kfold_cross_validation import double_Kfold_CV

def print_score(mean, std):
    for key in mean.keys():
        print(key, "\t", mean[key], "+\-", std[key])

x_train, y_train, x_test, y_test = get_monk(1)

hyperparameters = load_hparams("nn")
grid_search = RandomGridSearch(hyperparameters)
#grid_search.run_search(x_train, y_train, verbose=False)
train_mean, train_std, val_mean, val_std, test_mean, test_std = double_Kfold_CV(x_train, y_train, grid_search, 5, verbose=False)

print("------ Train scores: ------ ")
print_score(train_mean, train_std)
print("------ Validation scores: ------ ")
print_score(val_mean, val_std)
print("------ Test scores: ------ ")
print_score(test_mean, test_std)

"""

