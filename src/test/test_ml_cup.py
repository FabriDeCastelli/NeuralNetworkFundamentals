from sklearn.model_selection import train_test_split
from src.main.dataset_handler import get_cup_dataset
from src.main.models.layers.dense import Dense
from src.main.models.model import Model

x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# --------- Our implementation ------------
model = Model()
model.add(Dense(10, 16, activation="relu"))
model.add(Dense(16, 8, activation="tanh"))
model.add(Dense(8, 3))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['RootMeanSquaredError'])

model.summary()

model.fit(x_train, y_train, 500, 32, False)

errors = model.evaluate(x_test, y_test)
print(errors)


# --------- Keras ------------
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=10, activation='relu'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(3, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['RootMeanSquaredError'])

model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=0)

# model.summary()

errors = model.evaluate(x_test, y_test)
print(errors)
