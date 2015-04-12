import numpy as np
import cv2
import matplotlib
import os

# Number of bins
bin_n = 16 

svm_params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = 0.001, gamma = 0.001, degree = 3, coef0 = 0)

#######################################################################################################################################################################
# In this section we set up the required functions for training and testing
#######################################################################################################################################################################

# Function that computes the Histogram of Oriented Gradients for a given image
# Input: An image, which could be either grayscale or color
# Output: A 64-element array which represents the feature vector
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    # Quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))    
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    # Hist is a 64 bit vector
    hist = np.hstack(hists)
    return hist

# Function that reads in images from a folder, converts them to float32, converts them to gray, then adds them to an
# array
# Input: A path to the folder and the number of images in that folder
# Output: An array containing the converted images
def read_images(path, numImages):
    # Initialize an empty array
    images = [None] * numImages;
    listing = os.listdir(path);
    index = 0;
    for file in listing:
        # Read each image, resize it, convert it to float32, then to grayscale. Finally add it to the array
        im = cv2.imread(path + '\\' + file);
        im = cv2.resize(im, (75, 75))
        #im = im.astype(np.float32);
        im = cv2.GaussianBlur(im,(3, 3), 0);
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
        images[index] = im;
        index += 1;
    return images

# Function that creates the labels for training
# Input: Number of negative and positive images
# Output: An array containing only zeros and ones, where the number of zeros is equal to the number of negative images, and the number of ones is equal to the number
# of positive images. The zeros are always in the beginning of the array
def create_responses(numNegImages, numPosImages):
    # In case the user inputs a negative number of images
    if (numNegImages <= 0 or numPosImages <= 0):
        print('Error: The number of positive and negative images must be greater than zero')
        return
    # Case where we have the same number of positive and negative images
    elif (numNegImages == numPosImages):
        responses = np.float32(np.repeat(np.arange(2),numNegImages)[:,np.newaxis]);
        return responses
    # Case where the number of negative images is greater than the number of positive images
    elif (numNegImages > numPosImages):
        responses = np.float32(np.repeat(np.arange(2),numPosImages)[:,np.newaxis]);
        for i in range(numNegImages - numPosImages):
            responses = np.insert(responses, 0, np.float32(np.array([0])), axis = 0);
        return responses
    # Case where the number of positive images is greater than the number of negative images
    else:
        responses = np.float32(np.repeat(np.arange(2),numNegImages)[:,np.newaxis]);
        for i in range(numPosImages - numNegImages):
            responses = np.insert(responses, len(responses), np.float32(np.array([1])), axis = 0)
        return responses

# Function that computes the HOG for every images in an array
# Input: An array of images
# Output: A numpy array of size number_of_images * 64
def create_hog(images):
    hogData = np.zeros((len(images), 64));
    for i in range(len(images)):
        hogData[i] = hog(images[i]);
    return hogData

#######################################################################################################################################################################
# In this section we set up the training of the algorithm
#######################################################################################################################################################################

# Folder paths for day and night
posTrainingPath = 'C:\Users\Owner\Documents\ELEC 494\New Training Data\Archive\DPTrain_1500';
negTrainingPath = 'C:\Users\Owner\Documents\ELEC 494\New Training Data\Archive\DNTrain_1500';

# Number of positive and negative images
numPosImages = 1500;
numNegImages = 1500;

# Creating the labels
responses = create_responses(numNegImages, numPosImages);

# Reading in the images
posImages = read_images(posTrainingPath, numPosImages);
negImages = read_images(negTrainingPath, numNegImages);

# Creating the feature vectors for the training images
posHog = create_hog(posImages);
negHog = create_hog(negImages);
trainingData = np.vstack((negHog, posHog));
trainingData = trainingData.astype(np.float32);

# Initializing SVM
svm = cv2.SVM();
svm.train(trainingData, responses, params=svm_params);
svm.save('svm_data.dat');
