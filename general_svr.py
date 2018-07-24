#!/usr/bin/env python
"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run(linear=True):
    """
    To run
    :param linear: True if kernel to be used is linear, False for RBF
    :return:
    """
    dataset = load_boston()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.DataFrame(dataset.target)
    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle=True, random_state=28)
    if linear:
        model = SVR(kernel='linear')
        model = model.fit(X_train, y_train)
        plt.title('Linear')
    else:
        # parameters = {'C': [1, 1000], 'gamma': [0.00001, 1}
        # clf = GridSearchCV(svr, parameters)
        # clf.fit(X_train, y_train)
        # print(clf.cv_results_)
        # model = clf.best_estimator_
        model = SVR(kernel='rbf', C=200, gamma=0.0001)
        model = model.fit(X_train, y_train)
        plt.title('RBF')

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    plt.plot(y_test, color='green', label='Original')
    plt.plot(y_test_pred, color='blue', label='Predicted')
    plt.legend()
    plt.show()
    print('MSE TRAIN = ' + str(mse(y_train, y_train_pred)))
    print('MSE TEST = ' + str(mse(y_test, y_test_pred)))


run(linear=True)