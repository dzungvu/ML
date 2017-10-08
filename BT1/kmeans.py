# Kmeans with 2 gaussians
# Vu The Dung 
# MSSV: 14520205
# Day update: 10/04/2017
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(11)

means = [[2, 2], [8, 3]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1))
K = 2

original_label = np.asarray([0]*N + [1]*N)

cluster = KMeans(n_clusters=2, random_state=0)
result = cluster.fit_predict(X)

centers = cluster.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=result)
plt.scatter(centers[:, 0], centers[:, 1])
plt.title("Kmeans")
plt.show()
