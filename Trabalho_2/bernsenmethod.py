import numpy as np 
import math as mp
import matplotlib.pyplot as plt
import cv2

def Bernsen_Thresholding(image,hist_name):

	"""
	Given a T limiar value, thats it is calculated as:

	 	T = (zmax - zmin)/2 -> where zmax and zmin are the maximun and maninum gray levels values, respectively, in a n x n neighborhood centered in (x,y)

	If the pixel p (x,y) intensity is bigger the T, the pixel will be considered a point of an object (1). But if is equal or lower then T, it
	will be considered part of the background of the image (0).

			--- For this assignment will be use a 3 X 3 neighborhood

	"""

	#Copy the original image and padding it for work with the images edges. In the numpy.pad() function, lines/colums with zeros will be added
	image_copy = image.copy()
	image_copy = np.pad(image, (1,1), 'constant')
	img_thresholded = image_copy.copy()

	#Acess each pixel (x,y) of the image an than applying the thresholing
	for x in range(1,image_copy.shape[0]-1):
		for y in range(1,image_copy.shape[1]-1):

			#Save in a numpay array all the pixels in the 3 x 3 neighborhood of the pixel (x,y)
			neighborhood = np.array([image_copy[x,y],image_copy[x,y-1], image_copy[x,y+1], image_copy[x-1,y-1], image_copy[x+1,y+1], image_copy[x+1,y], image_copy[x-1,y], image_copy[x-1,y+1], image_copy[x+1,y-1]])
			
			#Order the pixel in crescent order, to obtain the maximum and maninum gray levels values
			neighborhood = np.sort(neighborhood)		

			#Find the minimun and the maximun intensity of the neighborhood
			zmin = neighborhood[0]
			zmax = neighborhood[len(neighborhood) - 1]

			#Calculate Limiar T

			T = (int(zmax) + int(zmin))/2

			#Threshold the image given the T
			if(image_copy[x,y] <= T):
				img_thresholded[x,y] = 255

			else:
				img_thresholded[x,y] = 0

	#Image Histogram
	#Calculate image histogram 
	hist = cv2.calcHist([img_thresholded],[0],None,[256],[0,256])
	#PLot the histogram grafic
	plt.hist(img_thresholded.ravel(),256,[0,256])
	plt.title('Bernsen Method Histogram')
	plt.savefig('output/Bernsen/Histogram/Histogram_' + hist_name)

	return img_thresholded




