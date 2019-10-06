import numpy as np 
import math as mp
import matplotlib.pyplot as plt
import cv2

def Contrast_Thresholding(image,hist_name):
	"""

	 If a pixel(x,y) with a neighboorhood of n x n pixels, as the intensity is more closer with the maximun local intenisty, the pixel
	 it is considered a point of the background. Otherwise, if it is more closer with the minumun local intensity, it is considered a point
	 of an object.

	 	--- For this assignment will be use a n X n neighborhood passed in the parmeters of teh function
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
			min_value = neighborhood[0]
			max_value = neighborhood[len(neighborhood) - 1]

			if(abs((int(image_copy[x,y]) - min_value)) <= abs((int(image_copy[x,y]) - max_value))):
				img_thresholded[x,y] = 0

			elif(abs((int(image_copy[x,y]) - min_value)) > abs((int(image_copy[x,y]) - max_value))):
				img_thresholded[x,y] = 255

	#Image Histogram
	#Calculate image histogram 
	hist = cv2.calcHist([img_thresholded],[0],None,[256],[0,256])
	#PLot the histogram grafic
	plt.hist(img_thresholded.ravel(),256,[0,256])
	plt.title('Contrast Method Histogram')
	plt.savefig('output/Contrast/Histogram/Histogram_' + hist_name)

	return img_thresholded

	