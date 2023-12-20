
from src.main.dataset_handler import get_monk
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD

x_train, y_train, x_test, y_test = get_monk(1)

model = Model()
model.add(Dense(17, 4, activation="relu"))
model.add(Dense(4, 1, activation="sigmoid"))

optimizer = SGD(learning_rate=0.73, momentum=0.1)

model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["binary_accuracy"])

model.fit(x_train, y_train, epochs=3000, batch_size=12, verbose=True)

print(model.evaluate(x_test, y_test))