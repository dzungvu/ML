# Kmeans with 2 gaussians
# Vu The Dung 
# MSSV: 14520205
# Day update: 10/04/2017
import numpy as np
import matplotlib.pyplot as plt
import random

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

y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Kmeans")
plt.show()