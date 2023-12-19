from sklearn.model_selection import train_test_split

from dataset_handler import get_cup_dataset
from models.layers.dense import Dense
from models.model import Model
from activation import ReLu, Identity, Sigmoid
from optimizer import SGD
from loss import MSE
from metric import RootMeanSquaredError
from evaluation.grid_search import RandomGridSearch
from initializer import Random
from src.main.utils import load_hparams

x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

initializer = Random()
relu = ReLu()
identity = Identity()
sigmoid = Sigmoid()
sgd = SGD()
loss = MSE()
metrics = [RootMeanSquaredError()]

range = (-0.2, 0.2)

model = Model()
model.add(Dense(10, 16, initializer, range, relu))
model.add(Dense(16, 8, initializer, range, sigmoid))
model.add(Dense(8, 3, initializer, range, identity))

model.compile(sgd, loss, metrics)


model.summary()

model.fit(x_train, y_train, 500, 32, True)

errors = model.evaluate(x_test, y_test)
print(errors)

params = load_hparams("nn")
grid_search = RandomGridSearch(params)
# res = grid_search.run_search(x_train, y_train, True)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=10, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(3, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['RootMeanSquaredError'])

model.fit(x_train, y_train, epochs=500, batch_size=20, verbose=0)

# model.summary()

errors = model.evaluate(x_test, y_test)

print(errors)

