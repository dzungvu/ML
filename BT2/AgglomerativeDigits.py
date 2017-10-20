# Agglomerative clustering written digits
# Vu The Dung 
# MSSV: 14520205
# Day update: 20/10/2017

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))

X = np.reshape(data, (-1, 1))
connectivity = grid_to_graph(*data.shape)

ward = AgglomerativeClustering(n_clusters=n_digits, linkage='ward',
                               connectivity=connectivity)
ward.fit(X)
label = np.reshape(ward.labels_, data.shape)

plt.figure(figsize=(5, 5))
plt.imshow(data, cmap=plt.cm.gray)
for l in range(n_clusters):
    plt.contour(label == l, contours=1,
                colors=[plt.cm.spectral(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
plt.show()

