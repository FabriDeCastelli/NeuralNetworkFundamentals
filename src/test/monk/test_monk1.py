from sklearn.model_selection import train_test_split
from src.main.utilities.dataset_handler import get_monk
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
from src.main.regularizer import L2, L1
from src.main.utilities.utils import log_experiment, setup_experiment

x_train, y_train, x_test, y_test = get_monk(1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

model = Model()
model.add(Dense(17, 4, activation="relu", weight_initializer="glorot_uniform", bias_initializer="zeros"))
model.add(Dense(4, 1, activation="sigmoid", weight_initializer="glorot_uniform", bias_initializer="zeros"))

optimizer = SGD(learning_rate=0.4, momentum=0.6)
l2 = L2(0)

model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["binary_accuracy"])

_, history = model.fit(x_train, y_train, x_val, y_val, epochs=450, batch_size=8, verbose=False)

train_score = model.evaluate(x_train, y_train)
val_score = model.evaluate(x_val, y_val)
test_score = model.evaluate(x_test, y_test)
train_std = {}
val_std = {}
test_std = {}

for key in test_score.keys():
    train_std[key] = 0.0
    val_std[key] = 0.0
    test_std[key] = 0.0

log_experiment(setup_experiment("monk1.2"),model, train_score, train_std, val_score, val_std, test_score, test_std, [history])


print("------ Train scores: ------ ")
print(train_score)
print("------ Validation scores: ------ ")
print(val_score)
print("------ Test scores: ------ ")
print(test_score)
