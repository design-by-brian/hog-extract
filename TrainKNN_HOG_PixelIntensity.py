import os
import numpy as np
import cv2 as cv


def returnCroppedRegion(img, contour):
	x, y, w, h = cv.boundingRect(contour)
	return img[y:y + h, x:x + w]

directory = '/home/student/train/'
trainData = np.ones([1, 4180], dtype=np.float32)  # 3780 for HOG + 400 for Pixel Intensity
trainLabels = np.ones([1, 1], dtype=np.float32)

HOG = cv.HOGDescriptor()
for folder in range(0, 10): # loop through train folder
	folderData = np.empty([len(os.listdir(directory + str(folder))), 4180]) # create a dummy array of correct shape
	index = 0
	for file in os.listdir(directory + str(folder)):
		if file.endswith(".jpg"):
			image = cv.imread(directory + '/' + str(folder) + '/' + file) # read in image from train folder
			# perform a series of pre processing to match the output of digitExtract
			gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
			blur = cv.bilateralFilter(gray, 9, 20, 20)
			ret, threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
			kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
			imgDilated = cv.morphologyEx(threshold, cv.MORPH_DILATE, kernel)
			blur = cv.bilateralFilter(imgDilated, 5, 20, 20)
			edges = cv.Canny(blur, 400, 450)
			im2, contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
			contours.sort(reverse=True, key=lambda x: cv.contourArea(x, False))
			croppedDigit = returnCroppedRegion(threshold, contours[0])

			resized = cv.resize(threshold, (64, 128), interpolation=cv.INTER_AREA) # resize to suit hog
			hog = HOG.compute(resized) # compute the hod descriptor for the digit
			hog = np.transpose(hog) # flatten the hog array

			digitResized = cv.resize(croppedDigit, (20, 20), interpolation=cv.INTER_AREA)
			digitReshaped = digitResized.reshape(-1, 400).astype(np.float32) 

			combinedDescriptor = np.hstack((digitReshaped, hog)) # horizontally stach the arrays in to one long array

			folderData[index] = combinedDescriptor
			index = index + 1
	labels = np.full((len(os.listdir(directory + str(folder))), 1), folder, dtype=np.float32) # create an array of labels to match train
	trainLabels = np.vstack((trainLabels, labels)) # stack the labels vertically under the dummy array
	trainData = np.vstack((trainData, folderData)) # stack the training data vertically under the dummy array

trainData = np.delete(trainData, 0, 0) # remove the empty dummy
trainLabels = np.delete(trainLabels, 0, 0) # remove the empty dummy

trainData = np.float32(trainData) # convert to float 32

# Initiate kNN, train it on the training data
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, trainLabels)

# Save the data
np.savez('knn_hog.npz', train=trainData, train_labels=trainLabels)

