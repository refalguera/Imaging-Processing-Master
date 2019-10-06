import numpy as np 
import math as mp 
import matplotlib.pyplot as plt
import cv2

def Global_Thresholding(image,hist_name):

	"""
	Given a T limiar value, if the pixel p (x,y) intensity is bigger the T, the pixel will be considered a point of an object (1). But if is equal or lower then T, it
	will be considered part of the background of the image (0).

		--- In this assignment we will consider the limiar T valuing 128 (middle of the gray scale [0, 255] of the image);
	"""

	#Copy the original image
	img_thresholded = image.copy()

	#Define Limiar T
	T = 128

	#Acess each pixel (x,y) of the image an than applying the thresholing
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			#Apply the Thresholding to divide the image
			if(image[x,y] <= T):
				img_thresholded[x,y] = 255
			else:
				img_thresholded[x,y] = 0

	#Image Histogram
	#Calculate image histogram 
	hist = cv2.calcHist([img_thresholded],[0],None,[256],[0,256])
	#PLot the histogram grafic
	plt.hist(img_thresholded.ravel(),256,[0,256])
	plt.title('Global Method Histogram')
	plt.savefig('output/Global/Histogram/Histogram_' + hist_name)

	return img_thresholded

