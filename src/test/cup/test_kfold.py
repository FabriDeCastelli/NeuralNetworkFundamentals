from sklearn.model_selection import train_test_split

from src.main.dataset_handler import get_cup_dataset
from src.main.evaluation.kfold_cross_validation import Kfold_CV
from src.main.models.layers.dense import Dense
from src.main.models.model import Model
from src.main.optimizer import SGD
from src.main.utils import plot_history

x_train, y_train, _ = get_cup_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

model = Model()
model.add(Dense(10, 64, activation="relu"))
model.add(Dense(64, 32, activation="relu"))
model.add(Dense(32, 16, activation="relu"))
model.add(Dense(16, 3))

optimizer = SGD(learning_rate=0.0005, momentum=0.7)

model.compile(optimizer=optimizer, loss='mean_euclidean_error_loss', metrics=['root_mean_squared_error'])


(train_score, val_score), model, histories = Kfold_CV(x_train, y_train, model, 7, 1000, 16, False)
    
for i, history in enumerate(histories):
    print("---History of fold: ", i+1)
    plot_history(history)

print("------ Train scores: ------ ")
print(train_score[0], train_score[1])
print("------ Validation scores: ------ ")
print(val_score[0], val_score[1])
print("------ Test scores: ------ ")
print(model.evaluate(x_test, y_test))
