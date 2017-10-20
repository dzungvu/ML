# Spectral clustering written digits
# Vu The Dung 
# MSSV: 14520205
# Day update: 20/10/2017

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import SpectralClustering
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

radius = 3
n_points = 8 * radius
METHOD = 'uniform'
array = []

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape

for image in lfw_people.images:
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp, normed=True, bins=256, range=(0, 255))
    array.append(hist)

reduced_data = PCA(n_components=2).fit_transform(array)
algo = SpectralClustering(n_clusters=10)
algo.fit(reduced_data)
lables = lfw_people.target
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=lables)
plt.show()
