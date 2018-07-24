"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt


def run():
    dataset = load_boston()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.DataFrame(dataset.target)
    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=28)
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    plt.plot(y_train, color='green', label='Original')
    plt.plot(y_train_pred, color='blue', label='Predicted')
    plt.title('Train')
    plt.legend()
    plt.show()

    plt.plot(y_test, color='green', label='Original')
    plt.plot(y_test_pred, color='blue', label='Predicted')
    plt.title('Test')
    plt.legend()
    plt.show()

    print(cross_val_score(regressor, X, y, cv=10))
    print('MSE TEST = ' + str(mse(y_test, y_test_pred)))


run()
