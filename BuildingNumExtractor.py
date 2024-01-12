import cv2 as cv
import numpy as np
import os


# Function Name: returnCroppedRegionBorder
# Inputs: img - image to be cropped, contour - contour defining the region
# Outputs: returns the cropped region with a slight border
def returnCroppedRegionBorder(img, contour):
    x, y, w, h = cv.boundingRect(contour)
    return img[y-5:y + h + 5, x-5:x + w + 5]


# Function Name: returnCroppedRegionMinimal
# Inputs: img - image to be cropped, contour - contour defining the region
# Outputs: returns the cropped region
def returnCroppedRegionMinimal(img, contour):
    x, y, w, h = cv.boundingRect(contour)
    return img[y:y + h, x:x + w]


# Function Name: sortContours
# Inputs: digitInfo - array containing [croppedDigit, x-value of digit center]
# Outputs: returns the x_value
# Purpose: Used as the key for sorting contours by the center x_value to arrange them from left to
# right as in the image.
def sortContours(digitInfo):
    return digitInfo[1]


# Function Name: normaliseImage
# Inputs: img - image to be normalised by size
# Outputs: returns the size normalised image
# Purpose: Keep images square so as to reduce the issue of scale and aspect ratio
# for the areaExtract function.
def normaliseImage(img):
    height = img.shape[0]
    width = img.shape[1]
    dimensions = np.array([width, height])
    minDim = dimensions.min()
    resized = cv.resize(img, (minDim, minDim), interpolation=cv.INTER_AREA)
    return resized


# Function Name: preProcessing
# Inputs: img - image to be processed
# Outputs: returns the processed image
# Purpose: process image to make number contours stand out best and remove noise from the image.
# --- Pipeline ---
# 1 bilateral blur to remove noise but maintain edges.
# 2 convert to gray colour space
# 3 use the CLAHE class to help remove lighting glare by normalising the image histogram
# 4 Apply a binary threshold using Otsu
# 5 Use a cross structure to dilate the edges and close and gaps caused by blurring.
# 6 Finally bilateral blur again to remove noise while maintaining edges for use in finding contours
def preProcessing(img):
    blur = cv.bilateralFilter(img, 7, 200, 200) # blur the image
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY) # convert the image to gray colour space
    clahe = cv.createCLAHE(clipLimit=1.6, tileGridSize=(14, 14))
    cl1 = clahe.apply(gray) # apply filter to remove glare (histogram equalisation), remove peaks
    ret, threshold = cv.threshold(cl1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # convert to a binary image
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)) # create a 3x3 cross kernel
    imgDilated = cv.morphologyEx(threshold, cv.MORPH_DILATE, kernel) # dilate the objects in the binary image
    blur = cv.bilateralFilter(imgDilated, 5, 200, 200) # blur image again
    return blur


# Function Name: areaExtract
# Inputs: img - image to be analysed and searched for a building number
# Outputs: Returns the cropped building number in binary and BGR color spaces along with
# the bounding box in an array.
# Purpose: Aim is to find a building number in a wild image. Image is resized and pre processed.
# Edges are found on the processed image and subsequently used for finding contours. Contour properties such as
# Area, centre, height and width are used to select contours most likely to be digits. The cropped
# area is found using the digits contours.
def areaExtract(img):
    resized = normaliseImage(img)
    copy = resized.copy()
    preProcessedImg = preProcessing(resized)
    edges = cv.Canny(preProcessedImg, 400, 450) # find edges in the image
    contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE) # find contours from the edge image
    contours = sorted(contours, key=lambda x: cv.contourArea(x, True), reverse=True) # sort the contours by the area, descending

    digitContours = []
    y_values = []
    x_values = []
    for digit in contours:
        x, y, w, h = cv.boundingRect(digit)
        boxArea = w * h # determine the area of the bounding box of the contour
        digitArea = cv.contourArea(digit, True) # determine the area of the closed contour
        if 100 < digitArea < 20000 and (h > w) and (0.2 * h < w < 0.8 * h) and h < 5 * w and (
                0.2 * boxArea < digitArea < 0.9 * boxArea): # digit needs to meet set of area, h and w ranges based on experimentation
            digitContours.append(digit) # add contour to digitContours if it meets dimension criteria
            y_values.append(cv.minAreaRect(digitContours[0])[0][1]) # add the y value of the contour centre
            x_values.append(cv.minAreaRect(digitContours[0])[0][0]) # add the x value of the contour centre

    centre_y_base = np.median(y_values) # find the median y centre
    centre_x_base = np.median(x_values) # find the mdian x centre
    areas = []
    digitContours_xyAdjusted = []
    for contour in digitContours:
        centre_y = cv.minAreaRect(contour)[0][1]
        centre_x = cv.minAreaRect(contour)[0][0]
        if centre_y_base * 0.9 < centre_y < centre_y_base * 1.1: # if the contour is closely in line with the median y value
            if centre_x_base * 0.3 < centre_x < centre_x_base * 2.5: # if the contour is within a range of the median x
                digitContours_xyAdjusted.append(contour) # add the contour to digitContours_xyAdjusted
                areas.append(cv.contourArea(contour, True)) # also add the area of the contour

    area_base = np.mean(areas) # determine the mean of the areas
    finalContours = []
    for contour in digitContours_xyAdjusted:
        digitArea = cv.contourArea(contour, True)
        if 0.5 * area_base < digitArea < 2.7 * area_base:
            finalContours.append(contour) # if the contour is close to the mean, add it to finalContours

    combinedDigits = np.concatenate(finalContours) # add all digit contours together to get the bounding box
    croppedAreaThreshold = returnCroppedRegionBorder(preProcessedImg, combinedDigits) # get the cropped region on threshold, used for extractDigit
    croppedAreaBGR = returnCroppedRegionBorder(copy, combinedDigits) # get the cropped region in BGR, used for 'detectedAreaX'
    x, y, w, h = cv.boundingRect(combinedDigits) # bounding box used for 'boundingBoxX'

    return croppedAreaThreshold, croppedAreaBGR, [x, y, w, h]


# Function Name: digitExtract
# Inputs: croppedAreaEdges - image to be analysed and searched for digits, croppedAreaBGR -
# used for cropping and returning the cropped area with bounding boxes.
# Outputs: Returns digit contours sorted from left to right as they appear in the image, returns
# the cropped area in BGR color space
# Purpose: Aim is to find a building numbers in a the given cropped area.
def digitExtract(croppedThreshold, croppedAreaBGR):
    croppedAreaEdges = cv.Canny(croppedThreshold, 400, 450) # get edges on the cropped threshold area
    contours, hierarchy = cv.findContours(croppedAreaEdges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # find contours
	# should only be digit contours. Get external so avoid inner zeroes or fours.

    digitInfo = []
    for digit in contours:
        x_value = cv.minAreaRect(digit)[0][0] # determine x centre value of the digit contour
        digitArea = cv.contourArea(digit, True)
        if digitArea < -75: # filtering if an noise creeps through
            croppedRegion = returnCroppedRegionMinimal(croppedThreshold, digit) # get the cropped region for the digit to be classified
            if len(croppedRegion) > 5 and len(croppedRegion[0]) > 5: # last filter to remove sliver contours
                digitInfo.append([croppedRegion, x_value])
                x, y, w, h = cv.boundingRect(digit)
                cv.rectangle(croppedAreaBGR, (x, y), (x + w, y + h), (0, 0, 255), 1)

    digitInfo.sort(key=sortContours) # sort the contours by their centre_x value, so position left to right. Building number reads left to
	# right

    return digitInfo, croppedAreaBGR


directory = '/home/student/test'
file = 0
HOG = cv.HOGDescriptor()
for entry in os.scandir(directory):
    if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file(): # if the file ends with .jpg or .png
        image = cv.imread(entry.path) # read the image and store as 'image'
        croppedAreaEdges, croppedAreaBGR, boundingBox = areaExtract(image) # call areaExtract, finds the cropped digit areas, edges
	# and BGR. This is simply to stop digitExtract from having to perform processing again on the BGR image to find contours.
        imageDigits, croppedAreaBGRContours = digitExtract(croppedAreaEdges, croppedAreaBGR) # find the digit contours from teh cropped area
        buildingNumber = ''
        for image in imageDigits: # loop through the digits found in each image
            digitResized = cv.resize(image[0], (20, 20), interpolation=cv.INTER_AREA) # normalise the image to 20x20
            digitReshaped = digitResized.reshape(-1, 400).astype(np.float32) # flatten the image to 1x400
            hog = HOG.compute(cv.resize(image[0], (64, 128), interpolation=cv.INTER_AREA)) # compute the hog descriptor for the digit
            hog = np.transpose(hog) # flatten the hog descriptor

            combinedDescriptor = np.hstack((digitReshaped, hog)) # match digit to be classified with training format

            knn = cv.ml.KNearest_create() # create the knn object

	    # train on data created from hog and pixel intensities
            with np.load('knn_hog.npz') as data:
                train = data['train']
                train_labels = data['train_labels']

                knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

            ret, result, neighbours, dist = knn.findNearest(combinedDescriptor, k=3) # classify the digit using the trained knn and k=3

            buildingNumber = buildingNumber + str(int(ret)) # add numbers to the string building number
        file = file + 1

    f = open("/home/student/smith_brian_19463540/output/House" + str(file) + ".txt", "w")
    f.write("Building Number(" + str(file) + ") : " + buildingNumber)
    f.close()

    f = open("/home/student/smith_brian_19463540/output/BoundingBox" + str(file) + ".txt", "w")
    f.write(str(boundingBox[0]) + ',' + str(boundingBox[1]) + ',' + str(boundingBox[2]) + ',' + str(boundingBox[3]))
    f.close()

    cv.imwrite("/home/student/smith_brian_19463540/output/DetectedArea" + str(file) + ".jpg", croppedAreaBGRContours)
