from sklearn.model_selection import train_test_split

from src.main.callback import EarlyStopping
from src.main.utilities.dataset_handler import get_cup_dataset
from src.main.loss import MEE
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
import tensorflow as tf

from src.main.utilities.utils import plot_history, predictions_to_csv

x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# --------- Our implementation ------------
model = Model()
model.add(Dense(10, 64, activation="relu"))
model.add(Dense(64, 32, activation="relu"))
model.add(Dense(32, 16, activation="relu"))
model.add(Dense(16, 3))

optimizer = SGD(learning_rate=0.0006, momentum=0.75)
early_stopping = EarlyStopping(monitor='mean_squared_error', start_from_epoch=7000, patience=90, restore_best_weights=True)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['root_mean_squared_error', 'mean_euclidean_error'])

model.summary()

model, history = model.fit(x_train, y_train, x_test, y_test, epochs=15000, batch_size=12, verbose=True)
plot_history(history)
errors = model.evaluate(x_test, y_test)
print(errors)

predictions = model.predict(x_test)
predictions_to_csv(predictions)

# --------- Keras ------------
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import SGD

model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3))


def mean_euclidean_error(y_true, y_pred):
    return tf.reduce_mean(tf.norm(y_true - y_pred, axis=1))


sgd = SGD(learning_rate=0.0005, momentum=0.75)

model.compile(loss="mean_squared_error", optimizer=sgd,  metrics=['RootMeanSquaredError', mean_euclidean_error])

# model.fit(x_train, y_train, epochs=12000, batch_size=12, verbose=0)

# model.summary()

errors = model.evaluate(x_test, y_test)
print(errors)

