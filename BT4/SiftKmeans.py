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
descriptors = np.array([])
for fileName in glob.glob(os.path.join(dataPath, '*.jpg')):
    pic = cv2.imread(fileName)
    kp, des = cv2.SIFT().detectAndCompute(pic, None)
    descriptors = np.append(descriptors, des)

desc = np.reshape(descriptors, (len(descriptors)/128, 128))
desc = np.float32(desc)

reduced_data = PCA(n_components=2).fit_transform(desc)
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
kmeans.fit_predict(reduced_data)
labels = kmeans.labels_
convertArray = PCA(n_components=2)
dataArray = convertArray.fit_transform(reduced_data)
plt.scatter(dataArray[:, 0], dataArray[:, 1], c=labels)
plt.show()