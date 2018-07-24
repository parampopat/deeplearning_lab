#!/usr/bin/env python
"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


def normalize(a):
    """
    Calculates Z score
    :param a: Array to be normalized
    :return: Normalized Array
    """
    a = (a - np.mean(a)) / (np.std(a))
    return a


def inverse_normalize(a, st_deviation, the_mean):
    """
    Translates normalized values to original values
    :param a: Normalized array
    :param st_deviation: Standard deviation of original levels
    :param the_mean: Mean of original levels
    :return: Original Values
    """
    a = (a * st_deviation) + the_mean
    return a


def get_regressor(xtrain, ytrain, xtest, ytest):
    """
    Trains and returns the regressor
    :return: Trained Model
    """
    regressor = Sequential()
    regressor.add(Dense(26, input_dim=13, activation='linear'))
    regressor.add(Dense(2, activation='linear'))
    regressor.add(Dense(1, activation='linear'))
    regressor.compile(loss='mse', optimizer='adam')
    regressor.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=100, batch_size=30, verbose=2)
    return regressor


def train(to_plot=False):
    """
    Loads dataset and trains LSTM based regressor on it.
    :param stock: Name of Dataset
    :param to_plot: True if plots to be shown
    :return: Trained Model
    """

    # Load Dataset
    dataset = load_boston()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.DataFrame(dataset.target)
    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    regressor = get_regressor(X_train, y_train, X_test, y_test)

    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    if to_plot:
        plt.plot(y_train, color='green', label='original')
        plt.plot(y_train_pred, color='navy', label='predicted')
        plt.title('Train')
        plt.legend()
        plt.show()

        plt.plot(y_test, color='green', label='original')
        plt.plot(y_test_pred, color='navy', label='predicted')
        plt.title('Test')
        plt.legend()
        plt.show()

    print('MSE TRAIN = ' + str(mse(y_train, y_train_pred)))
    print('MSE TEST = ' + str(mse(y_test, y_test_pred)))
    print(mae(y_test, y_test_pred))
    return regressor


trained_regressor = train(to_plot=True)