import cv2 as cv
import numpy as np
from scipy import io
import pickle		

def _processHOG(num_samples: int, datatype: str, filename: str) -> None:
	data_mat = io.loadmat(filename) # load in training data matlab file as array
	data = data_mat['X']

	if num_samples > data.shape[3]:
		raise IndexError("Requested sample size exceeds dataset size")
	
	data = np.transpose(data, (3, 0, 1, 2)) #reshape so images are in rows
	labels = data_mat['y'][0:num_samples].astype(np.float32) # first n sample labels

	HOG = cv.HOGDescriptor()
	desc = np.ones([1, 3780], dtype=np.float32)  # 1x3780 for HOG Descriptor

	for image in data[0:num_samples,:,:,:]: # user-defined num samples.
		gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Convert to grayscale 1x32x32
		resized = cv.resize(gray_img, (64, 128), interpolation=cv.INTER_AREA) # resize to suit hog
		hog = HOG.compute(resized) # compute the hog descriptor for the digit
		hog = hog.reshape(-1, 3780).astype(np.float32) # flatten the hog array
		desc = np.vstack((desc, hog))
		
	desc = np.delete(desc, 0, 0) # remove the empty dummy
	data = {'desc':desc, 'labels':labels}

	print('HOG Descriptors Processed.')
	print('Descriptor Shape ', desc.shape)
	print('Descriptor Data Type ', desc.dtype)
	print('Labels Shape', labels.shape)
	print('Labels Data Type ', labels.dtype)

	with open('data/' + str(datatype) + '_DATA_' + str(num_samples) + '.pkl', 'wb') as f:
		pickle.dump(data, f)

def _processPI(num_samples: int, datatype: str, filename: str) -> None:
	data_mat = io.loadmat(filename) # load in training data matlab file as array
	data = data_mat['X']

	if num_samples > data.shape[3]:
		raise IndexError("Requested sample size exceeds dataset size")
	
	data = np.transpose(data, (3, 0, 1, 2)) #reshape so images are in rows
	labels = data_mat['y'][0:num_samples].astype(np.float32) # first n sample labels

	desc = np.ones([1,400], dtype=np.float32)  # 400 for Pixel Intensity

	for image in data[0:num_samples,:,:,:]: # user-defined num samples.
		# perform a series of pre processing to match the output of digitExtract
		gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		img = cv.bilateralFilter(gray_img, 5, 5, 5)
		ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
		img = cv.bitwise_not(img)
		kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
		img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)
		img = cv.bilateralFilter(img, 5, 5, 5)
		img = cv.Canny(img, 400, 450)
		contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
		# cv.drawContours(gray_img, contours, -1, (255,255,255), 1)
		contours = sorted(contours, key=lambda x: cv.contourArea(x, False), reverse=True)
		x, y, w, h = cv.boundingRect(contours[0])
		croppedDigit = img[y:y + h, x:x + w]
		digit_resized = cv.resize(croppedDigit, (20, 20), interpolation=cv.INTER_AREA)
		digit_reshaped = digit_resized.reshape(-1, 400).astype(np.float32)
		desc = np.vstack((desc, digit_reshaped))
		
	desc = np.delete(desc, 0, 0) # remove the empty dummy
	data = {'desc':desc, 'labels':labels}

	print('Pixel Intensity Descriptors Processed.')
	print('Descriptor Shape ', desc.shape)
	print('Descriptor Data Type ', desc.dtype)
	print('Labels Shape', labels.shape)
	print('Labels Data Type ', labels.dtype)

	with open('data/' + str(datatype) + '_DATA_' + str(num_samples) + '.pkl', 'wb') as f:
		pickle.dump(data, f)

if __name__ == "__main__":
	num_samples = 1000
	_processHOG(num_samples, 'TRAIN', 'data/train_32x32.mat')
	_processHOG(num_samples, 'TEST', 'data/test_32x32.mat')

	# _processPI(num_samples, 'TRAIN', 'data/train_32x32.mat')
	# _processPI(num_samples, 'TEST', 'data/test_32x32.mat')