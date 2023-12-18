import numpy as np  
from src.main.utils import shuffle_data, mean_std_scores, create_search_obejct
from src.main.models.model import Model

def double_Kfold_CV(X, y, grid_search = "grid_search", k = 5, compile_params=None):
    """
    function that allow to compute the double kflod cross validation
    
    :param X: dataset to train the model
    :param y: target of the dataset
    :param k: number of fold
    :param model: the model to be trained
    :param grid_search: a grid search string that allow to choose the grid search to be used
    :param compile_params: compile parameters for the model
    """
    
    X, y = shuffle_data(X, y)
    
    x_fold = np.array_split(X, k)
    y_fold = np.array_split(y, k)
    
    grid_search = create_search_obejct(grid_search)
    
    
    
    train_scores = []
    test_scores = []
    
    for i in range(k):
        x_train, x_test = x_fold[:i] + x_fold[i+1:], x_fold[i]
        y_train, y_test = y_fold[:i] + y_fold[i+1:], y_fold[i]   
        

        model = grid_search.run_seach(x_train, y_train, x_test, y_test, compile_params)
    
    pass


def Kfold_CV(X, y, model, k = 5, epochs = 500, batch_size = 20, verbose = False):
    """
    function that compute K-fold CV on a set
    
    :param: X to perform the CV
    :param: y: target of the dataset
    :param: model: the model to be trained
    :param: k: number of fold
    :param: epochs: number of epochs for the training
    :param: batch_size: size of the batch
    :param: verbose: verbose mode for fit function
    
    :return: mean and std of train scores as dictionary, mean and std of validation scores as dictionary 
    """
    
    x_fold = np.array_split(X, k)
    y_fold = np.array_split(y, k)
    
    train_scores = []
    val_scores = []
    
    for i in range(k):
        x_train, x_val = x_fold[:i] + x_fold[i+1:], x_fold[i]
        y_train, y_val = y_fold[:i] + y_fold[i+1:], y_fold[i]   
        
        model.fit(x_train, y_train, epochs, batch_size, verbose)
        
        train_scores[i] = model.evaluate(x_train, y_train)
        val_scores[i] = model.evaluate(x_val, y_val)
        
        
    return mean_std_scores(train_scores), mean_std_scores(val_scores) 
        
        
    
            
            
        

        
    
    
    
    
