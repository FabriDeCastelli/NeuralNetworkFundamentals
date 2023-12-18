from sklearn.model_selection import train_test_split
import numpy as np  
from src.main.utils import shuffle_data
from src.main.models.model import Model

def double_Kfold_CV(X, y, model, grid_search, k = 5, compile_params=None):
    """
    function that allow to compute the double kflod cross validation
    
    :param X: dataset to train the model
    :param y: target of the dataset
    :param k: number of fold
    :param model: the model to be trained
    :param grid_search: a grid search object 
    :param compile_params: compile parameters for the model
    """
  
    pass


def Kfold_CV(X, y, model, k = 5, epochs = 500, batch_size = 20, verbose = False):
    """
    function that compute K-fold CV on a set
    
    :param: X to peerform the CV
    :param: y: target of the dataset
    :param: k: number of fold
    :param: model: the model to be trained
    
    returm: the best model and it's score 
    """
    
    X, y = shuffle_data(X, y)
    
    x_fold = np.array_split(X, k)
    y_fold = np.array_split(y, k)
    
    best_model = None
    best_model_score = {"loss": -1, "RMSE": -1}
    
    for i in range(k):
        x_train, x_val = x_fold[:i] + x_fold[i+1:], x_fold[i]
        y_train, y_val = y_fold[:i] + y_fold[i+1:], y_fold[i]   
        
        model.fit(x_train, y_train, epochs, batch_size, verbose)
        
        
        
        

        
    
    
    
    
