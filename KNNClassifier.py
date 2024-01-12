import cv2 as cv
import pickle
import numpy as np

# Initiate kNN, train it on the training data
knn = cv.ml.KNearest_create()

with open('TRAIN_DATA_10000.pkl', 'rb') as f:
    train_pkl = pickle.load(f)
    train_desc = train_pkl['desc'] 
    train_labels = train_pkl['labels']
    knn.train(train_desc, cv.ml.ROW_SAMPLE, train_labels)

with open('TEST_DATA_10000.pkl', 'rb') as f:
    test_pkl = pickle.load(f)
    test_desc = test_pkl['desc'] 
    test_labels = test_pkl['labels']

    ret, result, neighbours, dist = knn.findNearest(test_desc,k=5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print( accuracy )