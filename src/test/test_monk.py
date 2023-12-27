
from src.main.dataset_handler import get_monk
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
from src.main.callback import EarlyStopping

x_train, y_train, x_test, y_test = get_monk(1)

model = Model()
model.add(Dense(17, 4, activation="relu"))
model.add(Dense(4, 1, activation="sigmoid"))

optimizer = SGD(learning_rate=0.73, momentum=0.1)
early_stopping = EarlyStopping(patience=20, start_from_epoch=200, delta=0.0001, monitor="loss", restore_best_weights=True, verbose=True)

model.compile(optimizer=optimizer, callback=early_stopping, loss="mean_squared_error", metrics=["binary_accuracy"])

model.fit(x_train, y_train, x_test, y_test, epochs=5001, batch_size=12, verbose=True)
