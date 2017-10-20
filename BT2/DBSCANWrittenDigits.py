# DBSCAN clustering written digits
# Vu The Dung 
# MSSV: 14520205
# Day update: 20/10/2017

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

labels = digits.target
reduced_data = PCA(n_components=2)
transformData = reduced_data.fit_transform(data)
db = DBSCAN(eps=0.3, min_samples=10).fit(transformData)

dataframe = pd.DataFrame({'labels':db.labels_, 'truth labels':digits.target})
cros = pd.crosstab(dataframe['labels'], dataframe['truth labels'])
print (cros)

plt.scatter(transformData[:, 0], transformData[:, 1], c=labels)
plt.show()