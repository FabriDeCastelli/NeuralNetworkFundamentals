import numpy as np

from src.main.utilities.utils import predictions_to_csv

predictions = np.array([[1, 2, 3], [4, 5, 6]])
predictions_to_csv(predictions, "abc.csv")