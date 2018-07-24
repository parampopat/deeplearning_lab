#!/usr/bin/env python
"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def run(neighbors=3):
    """
    To run
    :param neighbors: Number of neighbors
    :return:
    """
    iris = datasets.load_iris()
    even = [i for i in range(0, 150, 2)]
    X_train = iris.data[even, :]
    y_train = iris.target[even]
    odd = [i for i in range(1, 150, 2)]
    X_test = iris.data[odd, :]
    y_test = iris.target[odd]
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_test_pred, normalize=True))


run(5)
