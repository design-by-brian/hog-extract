import numpy as np
import cv2 as cv
import pickle

train_desc = []
train_labels = []

with open('HOGPixelIntensity_TRAIN_DATA.pkl', 'rb') as f:
    train_pkl = pickle.load(f)
    train_desc = train_pkl['train_desc'] 
    train_labels = train_pkl['train_labels']

# Initiate kNN, train it on the training data
knn = cv.ml.KNearest_create()
knn.train(train_desc, cv.ml.ROW_SAMPLE, train_labels)

with open('knn_hog.pkl', 'rb') as f:
    pickle.dump(knn, f)