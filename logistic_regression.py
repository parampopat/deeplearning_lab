#!/usr/bin/env python
"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)
X = dataset.values
y = pd.DataFrame(iris.target)
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=28)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75)
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], label=j)
# plt.show()

y_test_pred = classifier.predict(X_test)

x = []
for i in range(y_test.__len__()):
    x.append(i + 1)

plt.scatter(x, y_test, color='green', label='Original')
plt.title('Original')
plt.show()
plt.scatter(x, y_test_pred, color='blue', label='Predicted')
plt.title('Predicted')
plt.show()
print('Acc = ' + str(accuracy_score(y_test, y_test_pred)))
