import cv2 as cv
import numpy as np
from scipy import io
import pickle

def returnCroppedRegion(img, contour):
	x, y, w, h = cv.boundingRect(contour)
	return img[y:y + h, x:x + w]

train_data_mat = io.loadmat('data/train_32x32.mat') # load in training data matlab file as array

train_images = train_data_mat['X']
train_images = np.transpose(train_images, (3, 0, 1, 2)) #reshape so images are in rows

train_labels = train_data_mat['y'][0:1000].flatten().astype(np.float32) # first 1000 labels

train_desc = np.ones([4180,], dtype=np.float32)  # 3780 for HOG + 400 for Pixel Intensity

HOG = cv.HOGDescriptor()

for image in train_images[0:1000,:,:,:]: # first 1000 images
	org_img = image
	# perform a series of pre processing to match the output of digitExtract
	gray_img = cv.cvtColor(org_img, cv.COLOR_BGR2GRAY)
	img = cv.bilateralFilter(gray_img, 5, 5, 5)
	ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	img = cv.bitwise_not(img)
	kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
	img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)
	img = cv.bilateralFilter(img, 5, 5, 5)
	img = cv.Canny(img, 400, 450)
	contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
	cv.drawContours(gray_img, contours, -1, (255,255,255), 1)
	contours = sorted(contours, key=lambda x: cv.contourArea(x, False), reverse=True)
	croppedDigit = returnCroppedRegion(img, contours[0])

	resized = cv.resize(img, (64, 128), interpolation=cv.INTER_AREA) # resize to suit hog
	hog = HOG.compute(resized) # compute the hog descriptor for the digit
	hog = np.transpose(hog) # flatten the hog array

	digit_resized = cv.resize(croppedDigit, (20, 20), interpolation=cv.INTER_AREA)
	digit_reshaped = digit_resized.reshape(400).astype(np.float32)

	combinedDescriptor = np.hstack((digit_reshaped, hog)) # horizontally stack the arrays in to one long array
	train_desc = np.vstack((train_desc, combinedDescriptor))
	
train_desc = np.delete(train_desc, 0, 0) # remove the empty dummy
train_desc = np.float32(train_desc) # convert to float 32
print(train_desc.shape)

train_data = {'train_desc':train_desc, 'train_labels':train_labels}

with open('HOGPixelIntensity_TRAIN_DATA.pkl', 'wb') as f:
	pickle.dump(train_data, f)