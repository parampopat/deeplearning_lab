#!/usr/bin/env python
"""
__author__ = "Param Popat"
__version__ = "1"
__git__ = "https://github.com/parampopat/"
"""


from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def load_model(num_clust):
    """
    Returns a K Means clustering model
    :param num_clust: Number of clusters
    :return: K Means fitted model instance
    """
    return KMeans(n_clusters=num_clust, random_state=0).fit(X)


iris = datasets.load_iris()
X = iris.data
y = iris.target
kmeans = load_model(3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
labels = kmeans.labels_

ax.scatter3D(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), marker='*')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('K Means Clustered')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter3D(X[:, 3], X[:, 0], X[:, 2], c=y, marker='*')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Original Partitions')
plt.show()
