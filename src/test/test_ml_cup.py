import numpy as np
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from src.main.dataset_handler import get_cup_dataset
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
import tensorflow as tf

x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# --------- Our implementation ------------
model = Model()
model.add(Dense(10, 64, activation="relu"))
model.add(Dense(64, 32, activation="relu"))
model.add(Dense(32, 16, activation="tanh"))
model.add(Dense(16, 3))

optimizer = SGD(learning_rate=0.01, momentum=0.02)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['RootMeanSquaredError'])

model.summary()

model.fit(x_train, y_train, 400, 8, True)

errors = model.evaluate(x_test, y_test)
print(errors)

# --------- Keras ------------
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import SGD

model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(3))


def mean_euclidean_error(y_true, y_pred):
    return tf.reduce_mean(tf.norm(y_true - y_pred, axis=1))


sgd = SGD(learning_rate=0.001, momentum=0.02)

model.compile(loss="mean_squared_error", optimizer=sgd, metrics=['RootMeanSquaredError'])

callback = EarlyStopping(monitor='loss', patience=20)
model.fit(x_train, y_train, epochs=400, batch_size=8, verbose=2)

# model.summary()

errors = model.evaluate(x_test, y_test)
print(errors)

