# SpectralClustering WrittenDigits
# Vu The Dung 
# MSSV: 14520205
# Day update: 10/04/2017

import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.cluster import SpectralClustering
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))

labels = digits.target
sample_size = 300
reduced_data = PCA(n_components=2).fit_transform(data)

algo = SpectralClustering(n_clusters=n_digits)
algo.fit(reduced_data)

