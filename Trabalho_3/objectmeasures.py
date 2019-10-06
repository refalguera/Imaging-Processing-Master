import cv2
import numpy as np  
import imutils as imu
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

"""
	In this Assignment we are going to work with a RGB image (multispectral image), therefore, each pixel p(x,y) keep an array of size 3,
	that contains, in this format, the amount of Red, Green and Blue in (x,y).

	For delevelop all the tasks, it will be necessary to use the following libraries:
			- Numpy
			- Imutils
			- OpenCV
"""

def Color_Transformation(image):
	"""

		This function transform a RGB image of size M X N, into a Binary Image of size M X N, using a global thresholding. 
		Since the image has a white background and an color variation for the objects (expect white), the threshold T can be giving as:

									T = [255,255,255]  (white)

		Then, for each pixel p (x,y):

					g(x,y) = {	[255,255,255],	if p(x,y) = T
									
								[0,0,0]  if p(x,y) < T
												}
	"""
	#Copy the original image information
	image_transformed = image.copy()

	#Define the threshold T
	T = np.array((255,255,255),dtype = np.uint8)

	#Acess each pixel p(x,y) of the image
	for x in range(image_transformed.shape[0]):
		for y in range(image_transformed.shape[1]):

			#Apply the Thresholding
			#For threshold the image, use the np.array_equal from Numpy. This function compares the pixel value (which is an array of 3 elements)
			if(np.array_equal(image_transformed[x,y], T) == False):
				image_transformed[x,y] = np.array((0,0,0), dtype = np.uint8)

	return image_transformed


def Object_Contour_Extraction(image):
	"""
		This function gets the objects images outlines (contours) using the Canny outline operator.
		Since the image is binary, therefore, has pixels valuing 0 (objects) or 255 (background) , and we want to extrat the objects contours (pixel with 0 value)
		,it was decided to put the threshold T1 was valuing 120, and T2 was valuing 220.
			--- Obs: In the Canny method, T2 > T1. A point is considered part of the image edge if its magnitude is bigger than T2. His neighbors are edges too, if
				their values is bigger than T1.
	"""

	image_transformed = Color_Transformation(image)
	images_edges = cv2.Canny(image_transformed,120,220)

	return images_edges

def Object_Features_Extraction(image,hist_name):
	"""
		This function gets some Features of the objects in the images as: Centroid, Perimeter and Area.

			- The centroid of a shape is the arithmetic mean (i.e. the average) of all the points in a shape.
		We can find the center of the blob using moments in OpenCV. But first of all, we should know what exactly Image moment is all about.
		Image Moment is a particular weighted average of image pixel intensities, 
		with the help of which we can find some specific properties of an image, like radius, area, centroid,
		etc. To find the centroid of the image, we generally convert it to binary format and then find its center.

		The centroid formula is given by:
				Cx = M10/ M00
				Cy = M01/ M00

			 -  An object Perimeter is the sum of each point of the object edge. Therefore, we need to sum all the points related to the contour of the object.
		The OpenCV Library offers the arglenght() function that calculates the perimeter of a curve.

			 - An object area is a mathematical concept that can be defined as the amount of two-dimensional space, ie surface of this object.
		The OpenCV Library offers the countourArea() function that calculates the perimeter of a curve.

		For extract the centroid, perimeter and area, will be necessary to extract the objects contours, and by that calculte its center.
	"""
	#Gets image with oly the objects edges
	image_transformed = Object_Contour_Extraction(image)
	image_feature = image.copy()

	#After applying thresholding the shapes are represented as a black foreground on a white background.
	#Find the location of these black regions using contour detection.
	contours = cv2.findContours(image_transformed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = imu.grab_contours(contours)

	#Define perimeter and area variables.
	contour_perimeter =[]
	contour_area = []

	# print("Number of Regions: ", str(len(contours)))
	#Count each region of contour founded
	count = -1

	for i in contours:

		#CENTROID EXTRACTION
		# Compute the center of the contour
		M = cv2.moments(i)
		if(M["m00"] == 0):
			cX = 0
			cY = 0
		else:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])


		#PERIMETERS EXTRACTION
		#Compute the perimeter of the contour
		contour_perimeter.append(cv2.arcLength(i,True))

		#AREA EXTRACTION
		#Compute the area of the contour
		contour_area.append(cv2.contourArea(i))

		#Count each contour region finded
		count = count + 1

		#Put the centroid in each respective object of the image
		cv2.putText(image_feature,str(count), (cX, cY),
		cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)

		#Print the Area and Perimeter in each respective object of the image
		print("\nRegion " + str(count) + "		Area: " +str(contour_area[count]) + "		Perimeter: " + str(contour_perimeter[count]))

	Histogram_and_ObjectSize_Classification(contours,contour_area,hist_name)

	return image_feature

def Histogram_and_ObjectSize_Classification(contours, contour_area, hist_name):
	"""
		This function will classify objects according with their area. The classification base is:

		small object: area < 1500 pixels
		medium object: area ≥ 1500 pixels and area < 3000 pixels
		big object: area ≥ 3000
	"""

	#Define
	small = 0
	medium = 0
	big = 0

	#Object Classification
	for i in range(len(contour_area)):
		if(i < 1500):
			small = small +1
		elif(i >= 1500 and i < 3000):
			medium = medium + 1
		else:
			big = big + 1

	print("\n Number of Small Regions: " + str(small) + "\n Number of Medium Regions: " + str(medium) + "\n Number of Big Regions: " + str(big))
	#Histogram

	n_bins= 10
	# the histogram of the data
	plt.hist(contour_area, n_bins, facecolor='blue', alpha=0.5)

	plt.title('Object Area Histogram')
	plt.xlabel('Area')
	plt.ylabel('Number of Objects')
	plt.savefig('output/Object_Area_Histogram/Histogram_' + hist_name)





