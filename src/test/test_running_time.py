import argparse

from sklearn.model_selection import train_test_split
from src.main.callback import EarlyStopping
from src.main.utilities.dataset_handler import get_cup_dataset
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
import tensorflow as tf

parser = argparse.ArgumentParser(description='Command parser')

parser.add_argument('--num-redo', type=int, help='Number of redo')

args = parser.parse_args()

num_redo = args.num_redo

x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# --------- Our implementation ------------

model = Model()
model.add(Dense(10, 164, activation="relu"))
model.add(Dense(164, 82, activation="tanh"))
model.add(Dense(82, 3))

optimizer = SGD(learning_rate=0.0006, momentum=0.73)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['root_mean_squared_error', 'mean_euclidean_error'])


now = tf.timestamp()
for i in range(num_redo):

    model, history = model.fit(x_train, y_train, x_test, y_test, epochs=5000, batch_size=14, verbose=False)
    model.reset()

now = (tf.timestamp() - now).numpy()
print(f"Fitting time over {num_redo} runs:", now)
print(f"Average fitting time over {num_redo} runs:", now / num_redo)
