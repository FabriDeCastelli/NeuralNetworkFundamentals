from pathlib import Path
from typing import Any

import yaml
import numpy as np

def read_yaml(path: str | Path) -> dict[str, Any]:
    """Reads a file in .yaml format."""
    with open(path, "r") as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    return dictionary

def load_hparams(classifier: str) -> dict[str, Any]:
    """Loads the hyperparameters for a certain model."""
    return read_yaml(HPARAMS_ROOT / f"{classifier}.yaml")

def shuffle_data(X, y):
    """shuffle data inside the dataset"""
    num_samples = X.shape[0]
    shuffled_indices = np.random.permutation(num_samples)
    
    return X[shuffled_indices], y[shuffled_indices]

def initialize_score(model): #FORSE INUTILE(?)
    """initialize all the scores of the model to -1"""
    model_score = {"loss":float('inf')}
    for metric in model.metrics:
        model_score[metric.to_string()] = -1
          
    return model_score

def create_search_obejct(grid_type):
    grid_search = None
    params = load_hparams("nn")
    if(grid_type == "grid_search"):
        grid_search = GridSearch(params)
    elif(grid_type == "random_search"):    
        grid_search = Random(params)
    elif(grid_type == "nested_search"):
        grid_search = Nested(params)
    return grid_search

def mean_std_scores(scores):
    """compute the mean and the std of the scores returned as dictionaries"""
    
    mean_score = {}
    std_score = {}
    keys = scores[0].keys()
    scores = list(map(list, zip(*[[entry[key] for key in entry] for entry in scores])))
    result_dict = {key: values for key, values in zip(keys, scores)}
    
    for keys in result_dict:
        mean_score[keys] = np.array(result_dict[keys]).mean()
        std_score[keys] = np.array(result_dict[keys]).std()
    
    return mean_score, std_score
    
    