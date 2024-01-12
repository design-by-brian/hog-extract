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
	labels = data_mat['y'][0:num_samples].astype(np.float32) # first 1000 labels

	HOG = cv.HOGDescriptor()
	desc = np.ones([1, 4180], dtype=np.float32)  # 3780 for HOG + 400 for Pixel Intensity

	for image in data[0:num_samples,:,:,:]: # user-defined num samples.
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
		x, y, w, h = cv.boundingRect(contours[0])
		croppedDigit = img[y:y + h, x:x + w]

		resized = cv.resize(img, (64, 128), interpolation=cv.INTER_AREA) # resize to suit hog
		hog = HOG.compute(resized) # compute the hog descriptor for the digit
		hog = hog.reshape(-1, 3780).astype(np.float32) # flatten the hog array

		digit_resized = cv.resize(croppedDigit, (20, 20), interpolation=cv.INTER_AREA)
		digit_reshaped = digit_resized.reshape(-1, 400).astype(np.float32)

		combinedDescriptor = np.hstack((digit_reshaped, hog)) # horizontally stack the arrays in to one long array
		desc = np.vstack((desc, combinedDescriptor))
		
	desc = np.delete(desc, 0, 0) # remove the empty dummy
	data = {'desc':desc, 'labels':labels}

	print('Data Processed.')
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