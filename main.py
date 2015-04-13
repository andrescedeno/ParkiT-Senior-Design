import numpy as np
import cv2
import matplotlib
import os
import urllib2
import timeit
import time
import datetime

# Number of bins for HOG
bin_n = 16
minutes = 15
seconds = 10 #minutes*60
numCycles = 4
hours = [7,13,17,19,21]


#######################################################################################################################################################################
# In this section, we have all the functions required to handle the image processing
#######################################################################################################################################################################

# Function that takes extracts the coordinates of the spaces for which to test
# Input: Name of the text file which has the coordinates
# Output: A matrix of size nX4, where n is the number of spaces to test, and 4 represents the coordinates of each space
def get_coordinates(fileName):
    coordinates = [];
    with open(fileName) as f:
        for line in f:
            int_list = [int(i) for i in line.split()];
            coordinates.append(int_list);
    return coordinates

# Function that takes extracts the positions of the spaces for which to test. Positions are in matrix format
# Input: Name of the text file which has the positions
# Output: A matrix of size nX2, where n is the number of spaces to test, and 2 represents the x and y positions of each space
def get_positions(fileName):
    positions = [];
    with open(fileName) as f:
        for line in f:
            int_list = [int(i) for i in line.split()];
            positions.append(int_list);
    return positions

# Function that performs the cropping of the spaces to test
# Input: An image of the parking lot, and the matrix of coordinates
# Output: An array of cropped parking space
def get_crops(image, coordinates):
    crops = [None] * len(coordinates);
    for i in range(len(coordinates)):
        im = image[coordinates[i][0]:coordinates[i][1], coordinates[i][2]:coordinates[i][3]]; 
        im = cv2.resize(im, (75, 75))
        #im = cv2.GaussianBlur(im,(3, 3), 0);
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
        crops[i] = im;
    return crops

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

# Function that computes the HOG for every images in an array
# Input: An array of images
# Output: A numpy array of size number_of_images * 64
def create_hog(images):
    hogData = np.zeros((len(images), 64));
    for i in range(len(images)):
        hogData[i] = hog(images[i]);
    return hogData

def grab(imageName, write = True):
	'''Gets the image and saves as timestamp '''
	
	response = urllib2.urlopen('http://10.129.59.177/axis-cgi/jpg/image.cgi')
	if write:
		#print "Writing"
		with open(imageName, 'wb') as outfile:
			outfile.write(response.read())
	else: 
		print "Not Writing"
	return response


def timeTest(N):
	secs = timeit.timeit("grabImage.grab('TimeitTest.jpg')",setup="import grabImage",number=N)
	print secs
	print 1.0*secs/N
	
	
def getImageName():
	'''Gets the time stamp and other info that defines image name. '''
	t = time.time()
	timeStamp = datetime.datetime.fromtimestamp(t).strftime('%m%d%Y-%H%M%S') #time stamp
	fileName = 'WestLot ' + timeStamp +'.png' 
	return fileName

def loop():
	while True:
		t = time.time()
		timeStamp = datetime.datetime.fromtimestamp(t).strftime('%m%d%Y-%H%M%S') #time stamp
		h = time.strftime('%H')#the current hour
		fileName = 'WestLot ' + timeStamp +'.png' 
		#for i in xrange(numCycles):
		if h in hours:
			grab(fileName)
			print fileName
			time.sleep(seconds)

# The parking spot class
class Spot:
    def __init__(self, x, y):
        self.row = x
        self.col = y

#######################################################################################################################################################################
# In this section we call all the functions required to obtain the results
#######################################################################################################################################################################

def main_func():
	
	#grab the image
	fileName = getImageName()
	grab(fileName)
	image = cv2.imread(fileName)
	
	# Need to add in Andres's code to grab an image
	coordinates = get_coordinates('coordinates.txt');
	positions = get_positions('positions.txt');
	crops = get_crops(image, coordinates);
	hogData = create_hog(crops);
	hogData = hogData.astype(np.float32);

	# Initialize SVM
	svm = cv2.SVM();
	svm.load('svm_data.dat');
	result = svm.predict_all(hogData);

	spaces_list = [];
	for i in range(len(result)):
		if (result[i][0] == 1):
			spaces_list.append(Spot(positions[i][0], positions[i][1]));
			cv2.circle(image, ((coordinates[i][2] + coordinates[i][3])/2, (coordinates[i][0] + coordinates[i][1])/2), 35, (0, 0, 255), 5);
	cv2.imwrite('Result.jpg', image);
	return spaces_list
            
#Run when main.py called
if __name__ == "__main__":
	main_func()

