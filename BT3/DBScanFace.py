# DBSCAN clustering face
# Vu The Dung 
# MSSV: 14520205
# Day update: 20/10/2017

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern

radius = 3
n_points = 8 * radius
METHOD = 'uniform'
array = []

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
lables = lfw_people.target

n_samples, h, w = lfw_people.images.shape

for image in lfw_people.images:
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp, normed=True, bins=256, range=(0, 255))
    array.append(hist)

reduced_data = PCA(n_components=2)
transformData = reduced_data.fit_transform(array)
db = DBSCAN(eps=0.3, min_samples=10).fit_predict(transformData)

plt.scatter(transformData[:, 0], transformData[:, 1], c=lables)
plt.show()