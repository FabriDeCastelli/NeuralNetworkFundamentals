import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import SGD
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.main.dataset_handler import get_cup_dataset

x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create a Sequential model
model = Sequential()

# Add the input layer
model.add(Dense(units=10, input_dim=10, activation='relu'))

# Define different units for each hidden layer
units_list = [20, 30, 40, 50, 40, 30, 20, 15, 12, 10, 8, 6, 4]

# Add the hidden layers with varying units
for units in units_list:
    model.add(Dense(units=units, activation='relu'))

# Add the output layer
model.add(Dense(units=3))


# Compile the model
def mean_euclidean_error(y_true, y_pred):
    return tf.reduce_mean(tf.norm(y_true - y_pred, axis=1))


sgd = SGD(learning_rate=0.001, momentum=0.9)

model.compile(loss=mean_euclidean_error, optimizer=sgd, metrics=['RootMeanSquaredError'])

# model.fit(x_train, y_train, epochs=10000, batch_size=x_train.shape[0], verbose=2)

# model.summary()

errors = model.evaluate(x_test, y_test)
print(errors)


x = np.ndarray(shape=(2, 10), dtype=float, order='F')
print(x)