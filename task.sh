#!/bin/sh
TRAINDATA=knn_hog.npz
if [ ! -f "$TRAINDATA" ]; then
	echo "No training data exists..."
	echo "Training KNN classifier."
	python3 TrainKNN_HOG_PixelIntensity.py
	echo "Extracting and classifying digits"
	python3 BuildingNumExtractor.py
	echo "Process complete..."
else
	echo "Training data exists."
	echo "Extracting and classifying digits"
	python3 BuildingNumExtractor.py
	echo "Process complete..."
fi
