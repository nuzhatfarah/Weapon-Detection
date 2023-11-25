# Weapon-Detection
This repository contains a Python implementation of a Weapon Detection System using image processing and machine learning techniques. The system is designed to analyze images and classify whether a weapon is present or not. The system follows a series of steps, including image preprocessing, feature extraction, clustering, and classification using Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Decision Trees.

Image Preprocessing:

The system reads images from a specified directory and resizes them to a standard size (224x224 pixels).
Converts the images to grayscale, applies edge detection using the Canny algorithm, and further thresholding to enhance weapon features.
Feature Extraction:

Utilizes K-Means clustering to identify patterns and group similar image regions together.
Extracts features by flattening the processed images into one-dimensional arrays, which serve as input data for the machine learning models.
Labeling:

Reads label information from corresponding text files. Each text file contains coordinates representing bounding boxes around weapons in the images.
Clustering:

Applies K-Means clustering to categorize images into two clusters based on their features. This step aims to group images with and without weapons.
Machine Learning Models:

Trains a Support Vector Machine (SVM) using linear kernel for classification.
Employs K-Nearest Neighbors (KNN) and  Decision Tree Classifie for different classification.

Evaluation Metrics:

Assesses the performance of the models using accuracy as the primary metric.
Additionally, computes precision, recall, and F1 score for the SVM model.
Evaluates the K-Nearest Neighbors model using accuracy.

Visualization:

Displays a pair of random images from each cluster to visually represent the clustering results.

Usage:

Ensure the required libraries are installed (cv2, numpy, os, sklearn, matplotlib).
Set the images_path and labels_path variables to the appropriate directories.
Run the script to train and evaluate the weapon detection system.
