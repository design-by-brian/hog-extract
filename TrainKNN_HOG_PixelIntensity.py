import numpy as np
import cv2 as cv
import pickle

train_desc = []
train_labels = []

with open('HOGPixelIntensity_TRAIN_DATA.pkl', 'rb') as f:
    train_pkl = pickle.load(f)
    train_desc = train_pkl['train_desc']
    train_labels = train_pkl['train_labels']
    
print(train_desc.shape)
print(train_desc.dtype)

print(train_labels.shape)
print(train_labels.dtype)


# Initiate kNN, train it on the training data
knn = cv.ml.KNearest_create()
knn.train(train_desc, cv.ml.ROW_SAMPLE, train_labels)

# Save the data
np.savez('knn_hog.npz', train=train_desc, train_labels=train_labels)