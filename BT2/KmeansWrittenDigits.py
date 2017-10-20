# Kmeans clustering written digits
# Vu The Dung 
# MSSV: 14520205
# Day update: 20/10/2017

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))

labels = digits.target
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit_predict(reduced_data)

dataframe = pd.DataFrame({'labels':kmeans.labels_, 'truth labels':digits.target})
cros = pd.crosstab(dataframe['labels'], dataframe['truth labels'])
print (cros)


convertArray = PCA(n_components=2)
dataArray = convertArray.fit_transform(digits.data)
plt.scatter(dataArray[:, 0], dataArray[:, 1], c=labels)
plt.show()
