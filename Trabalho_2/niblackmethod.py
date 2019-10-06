import numpy as np 
import math as mp
import matplotlib.pyplot as plt
import cv2

def Niblack_Thresholding(image,hist_name):
	"""
	Given a T limiar value, thats it is calculated as:

	 	T (x, y) = μ(x, y) + k σ(x, y) -> where μ and σ are the local average and the standard deviation, respectively, in a n x n neighborhood centered in (x,y)]
	 		--- The value of k is used to adjust the edge fraction of the object to be considered as part of the object.

	If the pixel p (x,y) intensity is bigger the T, the pixel will be considered a point of an object (1). But if is equal or lower then T, it
	will be considered part of the background of the image (0).

			--- For this assignment will be use a 3 X 3 neighborhood and a K valuing 0.8
	"""

	#Copy the original image and padding it for work with the images edges. In the numpy.pad() function, lines/colums with zeros will be added
	image_copy = image.copy()
	image_copy = np.pad(image, (1,1), 'constant')
	img_thresholded = image_copy.copy()

	#Constant K valuing 0.8
	k = 0.8

	#Acess each pixel (x,y) of the image an than applying the thresholing
	for x in range(1,image_copy.shape[0]-1):
		for y in range(1,image_copy.shape[1]-1):

			#Save in a numpay array all the pixels in the 3 x 3 neighborhood of the pixel (x,y)
			neighborhood = np.array([image_copy[x,y],image_copy[x,y-1], image_copy[x,y+1], image_copy[x-1,y-1], image_copy[x+1,y+1], image_copy[x+1,y], image_copy[x-1,y], image_copy[x-1,y+1], image_copy[x+1,y-1]])
			
			#Calculate the local average and standard variation based on the pixel neighborhood
			local_average = np.mean(neighborhood)
			#For the standard deviation use the function provided by the numpy library -> np.std()
			standard_deviation = np.std(neighborhood)

			#Calculate Limiar T

			T = local_average + (k *standard_deviation)

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
	plt.title('Niblack Method Histogram')
	plt.savefig('output/Niblack/Histogram/NHistogram_'+hist_name)
	
	return img_thresholded
