# Kmeans clustering with Flowers data - using sift feature
# Vu The Dung 
# MSSV: 14520205
# Day update: 20/10/2017

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
import glob
import matplotlib.pyplot as plt

dataPath = 'Z:\Subjects\ML\ML\BT4\jpg'
hog = cv2.HOGDescriptor()
hogs = []
for fileName in glob.glob(os.path.join(dataPath, '*.jpg')):
    im = cv2.imread(fileName)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h = hog.compute(image_gray)
    hogs.append(h)

reduced_data = PCA(n_components=2).fit_transform(hogs)
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
kmeans.fit_predict(reduced_data)
labels = kmeans.labels_
convertArray = PCA(n_components=2)
dataArray = convertArray.fit_transform(reduced_data)
plt.scatter(dataArray[:, 0], dataArray[:, 1], c=labels)
plt.show()