#!/usr/bin/env python
"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def run(linear=False):
    iris = datasets.load_iris()
    dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = dataset.values
    y = pd.DataFrame(iris.target)
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle=True, random_state=28)

    # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, parameters)
    # clf.fit(iris.data, iris.target)
    # model = clf.best_estimator_
    if linear:
        model = svm.SVC(C=6.0, kernel='linear')
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        print('Linear')
    else:
        model = svm.SVC(C=10.0, kernel='rbf', gamma=0.01)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        print('RBF')

    x = []
    for i in range(y_test.__len__()):
        x.append(i+1)

    plt.scatter(x, y_test, color='green', label='Original')
    plt.title('Original')
    plt.show()
    plt.scatter(x, y_test_pred, color='blue', label='Predicted')
    plt.title('Predicted')
    plt.show()
    print('Acc = ' + str(accuracy_score(y_test, y_test_pred)))


run(False)
