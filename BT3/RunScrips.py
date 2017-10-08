# Kmeans with fade dataset
# Vu The Dung 
# MSSV: 14520205
# Day update: 08/10/2017

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

dataPath = 'Z:\Subjects\ML\ML\BT3\Data\lfw_home\lfw_funneled'
savePath = os.path.join(dataPath, 'TrainData.npy')
radius = 3
n_points = 8 * radius
data = np.load(savePath)
array = []
count = 0

array = []

for lbp in data:
    hist, _ = np.histogram(lbp, normed=True, bins=n_points + 2, range=(0, n_points + 2))
    array.append(hist)
    count += 1
print array

reduced_data = PCA(n_components=2).fit_transform(array)
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
kmeans.fit_predict(reduced_data)
labels = kmeans.labels_
convertArray = PCA(n_components=2)
dataArray = convertArray.fit_transform(reduced_data)
plt.scatter(dataArray[:, 0], dataArray[:, 1], c=labels)
plt.show()

