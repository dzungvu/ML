# Train lbp feature
# Vu The Dung 
# MSSV: 14520205
# Day update: 08/10/2017

from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern

import numpy as np
import os
import glob
import cv2


people = fetch_lfw_people(data_home='BT3\Data', min_faces_per_person=70, resize=0.4)

dataPath = 'Z:\Subjects\ML\ML\BT3\Data\lfw_home\lfw_funneled'
savePath = os.path.join(dataPath, 'TrainData.npy')
listDir = next(os.walk(dataPath))[1]
countImage = 0;
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

for itemFolder in listDir:
    path = dataPath + '\\' + itemFolder
    for fileName in glob.glob(os.path.join(path, '*.jpg')):
        print (fileName)
        countImage += 1
        image = cv2.imread(fileName)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(imageGray, n_points, radius, METHOD)
        np.save(savePath, lbp)

print 'Done'