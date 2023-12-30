from src.main.dataset_handler import get_monk
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
from src.main.regularizer import L2, L1
from src.main.utils import log_experiment, setup_experiment

x_train, y_train, x_test, y_test = get_monk(3)

model = Model()
model.add(Dense(17, 15, activation="relu", weight_initializer="glorot_uniform", bias_initializer="zeros"))
model.add(Dense(15, 1, activation="sigmoid", weight_initializer="glorot_uniform", bias_initializer="zeros"))

optimizer = SGD(learning_rate=0.8, momentum=0.8)
l2 = L2(0.0023)

model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["binary_accuracy"])

_, history = model.fit(x_train, y_train, x_test, y_test, epochs=500, batch_size=x_train.shape[0], verbose=False)

train_score = model.evaluate(x_train, y_train)
val_score = model.evaluate(x_test, y_test)
test_score = model.evaluate(x_test, y_test)
train_std = {}
val_std = {}
test_std = {}

for key in test_score.keys():
    train_std[key] = 0.0
    val_std[key] = 0.0
    test_std[key] = 0.0

log_experiment(setup_experiment("monk3"),model, train_score, train_std, val_score, val_std, test_score, test_std, [history])


print("------ Train scores: ------ ")
print(train_score)
print("------ Validation scores: ------ ")
print(val_score)
print("------ Test scores: ------ ")
print(test_score)
