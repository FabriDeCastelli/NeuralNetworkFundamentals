from dataset_handler import get_monk
from models.layers.dense import Dense
from models.model import Model
from activation import ReLu, Identity, Sigmoid
from optimizer import SGD
from loss import MSE
from metric import Accuracy
from evaluation.grid_search import RandomGridSearch
from initializer import Random
from src.main.utils import load_hparams

x_train, y_train, x_test, y_test = get_monk(1)

initializer = Random()
relu = ReLu()
identity = Identity()
sigmoid = Sigmoid()
sgd = SGD(learning_rate=0.9)
loss = MSE()
metrics = [Accuracy()]

range = (-0.2, 0.2)

model = Model()
model.add(Dense(17, 4, initializer, range, sigmoid))
model.add(Dense(4, 2, initializer, range, sigmoid))

model.compile(sgd, loss, metrics)

# model.summary()
"""

model.fit(x_train, y_train, 500, 20, False)

errors = model.evaluate(x_test, y_test)

print(errors)
"""

params = load_hparams("nn")
grid_search = RandomGridSearch(params)
res = grid_search.run_search(x_train, y_train, True)


"""
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(17, input_dim=17, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=20, verbose=0)

errors = model.evaluate(x_test, y_test)

print(errors)"""
