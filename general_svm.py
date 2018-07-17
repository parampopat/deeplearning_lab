'''
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
'''

from sklearn import svm, datasets
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
iris = datasets.load_iris()
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle=True)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
model = clf.best_estimator_
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
plt.plot(y_test, color='green', label='Original')
plt.plot(y_test_pred, color='blue', label='Predicted')
plt.legend()
plt.show()
