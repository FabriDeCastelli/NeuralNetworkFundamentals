import numpy as np

from src.main.utilities.utils import mean_std_scores, shuffle_data


def Kfold_CV(X, y, model, k=5, epochs=500, batch_size=20, verbose=False):
    """
    Function that compute K-fold CV on a set

    :param: X to perform the CV
    :param: y: target of the dataset
    :param: model: the model to be trained
    :param: k: number of fold
    :param: epochs: number of epochs for the training
    :param: batch_size: size of the batch
    :param: verbose: verbose mode for fit function

    :return: mean and std of train scores as dictionary, mean and std of validation scores as dictionary
    """

    X, y = shuffle_data(X, y) 
    x_fold = np.array_split(X, k)
    y_fold = np.array_split(y, k)

    train_scores = []
    val_scores = []
    histories = []

    for i in range(k):

        if verbose:
            print("--------- Fold: ----------", i + 1)

        x_train, x_val = np.concatenate(x_fold[:i] + x_fold[i + 1:]), x_fold[i]
        y_train, y_val = np.concatenate(y_fold[:i] + y_fold[i + 1:]), y_fold[i]

        model.reset()
        
        trained_model, history = model.fit(x_train, y_train, x_val, y_val, epochs, batch_size, verbose)
        
        train_scores.append(trained_model.evaluate(x_train, y_train))
        val_scores.append(trained_model.evaluate(x_val, y_val))
        histories.append(history)

    print("Finished K-fold CV")
    return (mean_std_scores(train_scores), mean_std_scores(val_scores)), histories

