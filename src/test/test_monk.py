
from src.main.dataset_handler import get_monk
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
from src.main.regularizer import L2
from src.main.utils import plot_history

x_train, y_train, x_test, y_test = get_monk(1)

model = Model()
model.add(Dense(17, 4, activation="relu", weight_initializer="random_normal", bias_initializer="random_normal"))
model.add(Dense(4, 1, activation="sigmoid", weight_initializer="random_normal", bias_initializer="random_normal"))

optimizer = SGD(learning_rate=0.5, momentum=0.1)
l2 = L2(0.002)

model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["binary_accuracy"])

_, history = model.fit(x_train, y_train, x_test, y_test, epochs=1000, batch_size=16, verbose=True)

plot_history(history)
