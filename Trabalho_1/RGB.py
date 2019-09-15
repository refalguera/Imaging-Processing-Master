import cv2
import math as mp
import numpy as np

#Function that copys the elements of the original image to the center of new matrix
def copy_element(matrix,padding,copymatrix):

	matrix[padding:(matrix.shape[0] - padding),padding:(matrix.shape[1] - padding)] = copymatrix[:]

#Function that evaluets if the pixel value its on the limits, that is 0 or 255
#If the value is bigger than 255, the pixel assume the 255 value
#If is lower than 0, the pixel assume the 0 value
def value_boundary(value):
	if value > 255:
		value = 255
	if value < 0:
		value = 0

	return value

#Function that selects the scan direction
#If the activator is equal to 1, than ativate the ZigZag direction (when the line is even (multiple of 2)
# starts at the end to the beging of the matrix)
#If the activator is equal to 0, than ativate the Normal direction (when the line is impar (not multiple of 2)
# starts at the begining to the end of the matrix)
#The function also activates the flip mask activator value. If its the ZigZag scan direction, than flip_mask is 1 (the mask will flip). If its not than fli-mask is 0 won't flip.
def scan_direction(x,n,padding,activator):
	
	if(activator == 1):
	#ZigZag Order
		if (x%2) == 0:
			lmax = padding
			lmin = n - padding
		else:
			lmax = n - padding
			lmin = padding

		flip_mask = 1

	#Normal Order
	else:
		lmax = n-padding
		lmin = padding
		flip_mask = 0

	return lmax, lmin, flip_mask

#Floyd and Steinberg Techinique
#	3/16 f(x, y) 7/16
#	5/16  1/16

def Floyd_Stein(image,m,n,activator):

	#Create an error numpy array full of zeros, for save the error that will be calculated forwards.
	error = np.zeros(3)
	#Create an zero integer matrix of shape (a,b, k), of size N times bigger than the original image
	imgf = np.zeros(shape= (m+2,n+2,3)).astype(np.uint8)


	#Copy the values of the original image to the new matrix
	copy_element(imgf,1,image)

	#Create a coeficient mask matrix
	mask_coef = np.array([[1,	1,	7/16],
						  [3/16,5/16,1/16]])

	#Access each point (x,y) of the matrix
	for x in range(1,m-1):
		#Calculates the order of the loop, based on which scan direction was choosed.
		lmax,lmin,flip_mask = scan_direction(x,n,1,activator)

		for y in range(lmin,lmax):

			#Save the old values of the point (x,y) for each channel.
			old_blue_pixel = imgf[x,y][0]
			old_green_pixel = imgf[x,y][1]
			old_red_pixel = imgf[x,y][2]
	
			
			#Generates the new pixel value - 0 or 255 (White or Black) for each channel. for each channel.
			new_red_pixel = 255 * mp.floor(old_red_pixel/128.0)
			new_green_pixel = 255 * mp.floor(old_green_pixel/128.0)
			new_blue_pixel = 255 * mp.floor(old_blue_pixel/128.0)

			#Attaches the new pixel value to the point (x,y) for each channel
			imgf[x,y][0] = new_blue_pixel
			imgf[x,y][1] = new_green_pixel
			imgf[x,y][2] = new_red_pixel
	
			#Calculates the error between the new and the old value for each channel
			error[0] = old_blue_pixel - new_blue_pixel
			error[1] = old_green_pixel - new_green_pixel 
			error[2] = old_red_pixel - new_red_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask ==1):
				mask_coef = np.fliplr(mask_coef)

			#Apply error diffusion in each channel of all the adjacent pixels
			imgf[x+1,y][0] = value_boundary(imgf[x+1,y][0] + ((mask_coef[1,1]) * error[0]))
			imgf[x+1,y][1] = value_boundary(imgf[x+1,y][1] + ((mask_coef[1,1]) * error[1]))
			imgf[x+1,y][2] = value_boundary(imgf[x+1,y][2] + ((mask_coef[1,1]) * error[2]))

			imgf[x,y+1][0] = value_boundary(imgf[x,y+1][0] + ((mask_coef[0,2]) * error[0]))
			imgf[x,y+1][1] = value_boundary(imgf[x,y+1][1] + ((mask_coef[0,2]) * error[1]))
			imgf[x,y+1][2] = value_boundary(imgf[x,y+1][2] + ((mask_coef[0,2]) * error[2]))

			imgf[x+1,y-1][0] = value_boundary(imgf[x+1,y-1][0] + ((mask_coef[1,0]) * error[0]))
			imgf[x+1,y-1][1] = value_boundary(imgf[x+1,y-1][1] + ((mask_coef[1,0]) * error[1]))
			imgf[x+1,y-1][2] = value_boundary(imgf[x+1,y-1][2] + ((mask_coef[1,0]) * error[2]))

			imgf[x+1,y+1][0] = value_boundary(imgf[x+1,y+1][0] + ((mask_coef[1,2]) * error[0]))
			imgf[x+1,y+1][1] = value_boundary(imgf[x+1,y+1][1] + ((mask_coef[1,2]) * error[1]))
			imgf[x+1,y+1][2] = value_boundary(imgf[x+1,y+1][2] + ((mask_coef[1,2]) * error[2]))


	return imgf

#Burkes Techinique
#		  		f(x, y) 	8/32 	4/32
#2/32  4/32 	 8/32 		4/32 	2/32

def Burkes(image,m,n,activator):

	#Create an error numpy array full of zeros, for save the error that will be calculated forwards.
	error = np.zeros(3)

	#Create an zero integer matrix of shape (a,b, k), of size N times bigger than the original image
	imgf = np.zeros(shape=(m+4,n+4,3)).astype(np.uint8)
	#Copy the values of the original image to the new matrix
	copy_element(imgf,2,image)

	#Create a coeficient mask matrix
	mask_coef = np.array([[1,		1,	 	1,		8/32,	4/32],
						 [2/32,    4/32,   8/32,	4/32,	2/32]])

	#Access each point (x,y) of the matrix
	for x in range(2,m-2):
		#Calculates the order of the loop, based on which scan direction was choosed.
		lmax,lmin, flip_mask = scan_direction(x,m,2,activator)

		for y in range(lmin,lmax):
			
			#Save the old values of the point (x,y) for each channel.
			old_blue_pixel = imgf[x,y][0]
			old_green_pixel = imgf[x,y][1]
			old_red_pixel = imgf[x,y][2]
	
			
			#Generates the new pixel value - 0 or 255 (White or Black) for each channel. for each channel.
			new_red_pixel = 255 * mp.floor(old_red_pixel/128.0)
			new_green_pixel = 255 * mp.floor(old_green_pixel/128.0)
			new_blue_pixel = 255 * mp.floor(old_blue_pixel/128.0)

			#Attaches the new pixel value to the point (x,y) for each channel
			imgf[x,y][0] = new_blue_pixel
			imgf[x,y][1] = new_green_pixel
			imgf[x,y][2] = new_red_pixel
	
			#Calculates the error between the new and the old value for each channel
			error[0] = old_blue_pixel - new_blue_pixel
			error[1] = old_green_pixel - new_green_pixel 
			error[2] = old_red_pixel - new_red_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask ==1):
				mask_coef = np.fliplr(mask_coef)

			##Apply error diffusion in each channel of all the adjacent pixels
			imgf[x+1,y][0] = value_boundary(imgf[x+1,y][0] + ((mask_coef[1,2]) * error[0]))
			imgf[x+1,y][1] = value_boundary(imgf[x+1,y][1] + ((mask_coef[1,2]) * error[1]))
			imgf[x+1,y][2] = value_boundary(imgf[x+1,y][2] + ((mask_coef[1,2]) * error[2]))

			imgf[x,y+1][0] = value_boundary(imgf[x,y+1][0] + ((mask_coef[0,3]) * error[0]))
			imgf[x,y+1][1] = value_boundary(imgf[x,y+1][1] + ((mask_coef[0,3]) * error[1]))
			imgf[x,y+1][2] = value_boundary(imgf[x,y+1][2] + ((mask_coef[0,3]) * error[2]))

			imgf[x+1,y+1][0] = value_boundary(imgf[x+1,y+1][0] + ((mask_coef[1,3]) * error[0]))
			imgf[x+1,y+1][1] = value_boundary(imgf[x+1,y+1][1] + ((mask_coef[1,3]) * error[1]))
			imgf[x+1,y+1][2] = value_boundary(imgf[x+1,y+1][2] + ((mask_coef[1,3]) * error[2]))

			imgf[x,y+2][0] = value_boundary(imgf[x,y+2][0] + ((mask_coef[0,4]) * error[0]))
			imgf[x,y+2][1] = value_boundary(imgf[x,y+2][1] + ((mask_coef[0,4]) * error[1]))
			imgf[x,y+2][2] = value_boundary(imgf[x,y+2][2] + ((mask_coef[0,4]) * error[2]))

			imgf[x+1,y+2][0] = value_boundary(imgf[x+1,y+2][0] + ((mask_coef[1,4]) * error[0]))
			imgf[x+1,y+2][1] = value_boundary(imgf[x+1,y+2][1] + ((mask_coef[1,4]) * error[1]))
			imgf[x+1,y+2][2] = value_boundary(imgf[x+1,y+2][2] + ((mask_coef[1,4]) * error[2]))

			imgf[x+1,y-1][0] = value_boundary(imgf[x+1,y-1][0] + ((mask_coef[1,1]) * error[0]))
			imgf[x+1,y-1][1] = value_boundary(imgf[x+1,y-1][1] + ((mask_coef[1,1]) * error[1]))
			imgf[x+1,y-1][2] = value_boundary(imgf[x+1,y-1][2] + ((mask_coef[1,1]) * error[2]))

			imgf[x+1,y-2][0] = value_boundary(imgf[x+1,y-2][0] + ((mask_coef[1,0]) * error[0]))
			imgf[x+1,y-2][1] = value_boundary(imgf[x+1,y-2][1] + ((mask_coef[1,0]) * error[1]))
			imgf[x+1,y-2][2] = value_boundary(imgf[x+1,y-2][2] + ((mask_coef[1,0]) * error[2]))

	return imgf

#Stucki Techinique
#			f(x, y)	 8/42 	4/42
#2/42  4/42   8/42 	 4/42 	2/42
#1/42  2/42   4/42   2/42    1/4

def Stuki(image,m,n,activator):

	#Create an error numpy array full of zeros, for save the error that will be calculated forwards.
	error = np.zeros(3)

	#Create an zero integer matrix of shape (a,b, k), of size N times bigger than the original image
	imgf = np.zeros(shape=(m+4,n+4,3)).astype(np.uint8)
	#Copy the values of the original image to the new matrix
	copy_element(imgf,2,image)

	#Create a coeficient mask matrix
	mask_coef = np.array([[1,		1,	 	1,		8/42,	4/42],
						 [2/42,    4/42,   8/42,	4/42,	2/42],
						 [1/42,    2/42,   4/42,	2/42,	1/42]])

	#Access each point (x,y) of the matrix
	for x in range(2,m-2):
		#Calculates the order of the loop, based on which scan direction was choosed.
		lmax,lmin, flip_mask= scan_direction(x,n,2,activator)

		for y in range(lmin,lmax):
			
			#Save the old values of the point (x,y) for each channel.
			old_blue_pixel = imgf[x,y][0]
			old_green_pixel = imgf[x,y][1]
			old_red_pixel = imgf[x,y][2]
	
			
			#Generates the new pixel value - 0 or 255 (White or Black) for each channel. for each channel.
			new_red_pixel = 255 * mp.floor(old_red_pixel/128.0)
			new_green_pixel = 255 * mp.floor(old_green_pixel/128.0)
			new_blue_pixel = 255 * mp.floor(old_blue_pixel/128.0)

			#Attaches the new pixel value to the point (x,y) for each channel
			imgf[x,y][0] = new_blue_pixel
			imgf[x,y][1] = new_green_pixel
			imgf[x,y][2] = new_red_pixel
	
			#Calculates the error between the new and the old value for each channel
			error[0] = old_blue_pixel - new_blue_pixel
			error[1] = old_green_pixel - new_green_pixel 
			error[2] = old_red_pixel - new_red_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)


			##Apply error diffusion in each channel of all the adjacent pixels
			imgf[x,y+1][0] = value_boundary(imgf[x,y+1][0] + ((mask_coef[0,3]) * error[0]))
			imgf[x,y+1][1] = value_boundary(imgf[x,y+1][1] + ((mask_coef[0,3]) * error[1]))
			imgf[x,y+1][2] = value_boundary(imgf[x,y+1][2] + ((mask_coef[0,3]) * error[2]))

			imgf[x,y+2][0]= value_boundary(imgf[x,y+2][0]+ ((mask_coef[0,4]) * error[0]))
			imgf[x,y+2][1]= value_boundary(imgf[x,y+2][1]+ ((mask_coef[0,4]) * error[1]))
			imgf[x,y+2][2]= value_boundary(imgf[x,y+2][2]+ ((mask_coef[0,4]) * error[2]))

			imgf[x+1,y][0]= value_boundary(imgf[x+1,y][0]+ ((mask_coef[1,2]) * error[0]))
			imgf[x+1,y][1]= value_boundary(imgf[x+1,y][1]+ ((mask_coef[1,2]) * error[1]))
			imgf[x+1,y][2]= value_boundary(imgf[x+1,y][2]+ ((mask_coef[1,2]) * error[2]))

			imgf[x+2,y][0]= value_boundary(imgf[x+2,y][0]+ ((mask_coef[2,2]) * error[0]))
			imgf[x+2,y][1]= value_boundary(imgf[x+2,y][1]+ ((mask_coef[2,2]) * error[1]))
			imgf[x+2,y][2]= value_boundary(imgf[x+2,y][2]+ ((mask_coef[2,2]) * error[2]))

			imgf[x+1,y+1][0]= value_boundary(imgf[x+1,y+1][0]+ ((mask_coef[1,3]) * error[0]))
			imgf[x+1,y+1][1]= value_boundary(imgf[x+1,y+1][1]+ ((mask_coef[1,3]) * error[1]))
			imgf[x+1,y+1][2]= value_boundary(imgf[x+1,y+1][2]+ ((mask_coef[1,3]) * error[2]))

			imgf[x+1,y+2][0]= value_boundary(imgf[x+1,y+2][0]+ ((mask_coef[1,4]) * error[0]))
			imgf[x+1,y+2][1]= value_boundary(imgf[x+1,y+2][1]+ ((mask_coef[1,4]) * error[1]))
			imgf[x+1,y+2][2]= value_boundary(imgf[x+1,y+2][2]+ ((mask_coef[1,4]) * error[2]))

			imgf[x+1,y-1][0]= value_boundary(imgf[x+1,y-1][0]+ ((mask_coef[1,1]) * error[0]))
			imgf[x+1,y-1][1]= value_boundary(imgf[x+1,y-1][1]+ ((mask_coef[1,1]) * error[1]))
			imgf[x+1,y-1][2]= value_boundary(imgf[x+1,y-1][2]+ ((mask_coef[1,1]) * error[2]))

			imgf[x+1,y-2][0]= value_boundary(imgf[x+1,y-2][0]+ ((mask_coef[1,0]) * error[0]))
			imgf[x+1,y-2][1]= value_boundary(imgf[x+1,y-2][1]+ ((mask_coef[1,0]) * error[1]))
			imgf[x+1,y-2][2]= value_boundary(imgf[x+1,y-2][2]+ ((mask_coef[1,0]) * error[2]))

			imgf[x+2,y+1][0]= value_boundary(imgf[x+2,y+1][0]+ ((mask_coef[2,3]) * error[0]))
			imgf[x+2,y+1][1]= value_boundary(imgf[x+2,y+1][1]+ ((mask_coef[2,3]) * error[1]))
			imgf[x+2,y+1][2]= value_boundary(imgf[x+2,y+1][2]+ ((mask_coef[2,3]) * error[2]))

			imgf[x+2,y+2][0]= value_boundary(imgf[x+2,y+2][0]+ ((mask_coef[2,4]) * error[0]))
			imgf[x+2,y+2][1]= value_boundary(imgf[x+2,y+2][1]+ ((mask_coef[2,4]) * error[1]))
			imgf[x+2,y+2][2]= value_boundary(imgf[x+2,y+2][2]+ ((mask_coef[2,4]) * error[2]))

			imgf[x+2,y-1][0]= value_boundary(imgf[x+2,y-1][0]+ ((mask_coef[2,1]) * error[0]))
			imgf[x+2,y-1][1]= value_boundary(imgf[x+2,y-1][1]+ ((mask_coef[2,1]) * error[1]))
			imgf[x+2,y-1][2]= value_boundary(imgf[x+2,y-1][2]+ ((mask_coef[2,1]) * error[2]))

			imgf[x+2,y-2][0]= value_boundary(imgf[x+2,y-2][0]+ ((mask_coef[2,0]) * error[0]))
			imgf[x+2,y-2][1]= value_boundary(imgf[x+2,y-2][1]+ ((mask_coef[2,0]) * error[1]))
			imgf[x+2,y-2][2]= value_boundary(imgf[x+2,y-2][2] + ((mask_coef[2,0]) * error[2]))


	return imgf

#Sierra Techinique
#			f(x, y)	 5/32 	3/32
#2/32 4/32   5/32 	 4/32  	2/32
#	  2/32   3/32    2/32

def Sierra(image,m,n,activator):

	#Create an error numpy array full of zeros, for save the error that will be calculated forwards.
	error = np.zeros(3)

	#Create an zero integer matrix of shape (a,b, k), of size N times bigger than the original image
	imgf = np.zeros(shape=(m+4,n+4,3)).astype(np.uint8)
	#Copy the values of the original image to the new matrix
	copy_element(imgf,2,image)

	#Create a coeficient mask matrix
	mask_coef = np.array([[1,		1,	 	1,		5/32,	3/32],
						 [2/32,    4/32,   5/32,	4/32,	2/32],
						 [1, 	   2/32,   3/32,	2/32,	 1]])

	#Access each point (x,y) of the matrix
	for x in range(2,m-2):
		#Calculates the order of the loop, based on which scan direction was choosed.
		lmax,lmin,flip_mask = scan_direction(x,n,2,activator)

		for y in range(lmin,lmax):
			
			#Save the old values of the point (x,y) for each channel.
			old_blue_pixel = imgf[x,y][0]
			old_green_pixel = imgf[x,y][1]
			old_red_pixel = imgf[x,y][2]
	
			
			#Generates the new pixel value - 0 or 255 (White or Black) for each channel. for each channel.
			new_red_pixel = 255 * mp.floor(old_red_pixel/128.0)
			new_green_pixel = 255 * mp.floor(old_green_pixel/128.0)
			new_blue_pixel = 255 * mp.floor(old_blue_pixel/128.0)

			#Attaches the new pixel value to the point (x,y) for each channel
			imgf[x,y][0] = new_blue_pixel
			imgf[x,y][1] = new_green_pixel
			imgf[x,y][2] = new_red_pixel
	
			#Calculates the error between the new and the old value for each channel
			error[0] = old_blue_pixel - new_blue_pixel
			error[1] = old_green_pixel - new_green_pixel 
			error[2] = old_red_pixel - new_red_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)


			##Apply error diffusion in each channel of all the adjacent pixels
			imgf[x,y+1][0]= value_boundary(imgf[x,y+1][0]+ ((mask_coef[0,3]) * error[0]))
			imgf[x,y+1][1]= value_boundary(imgf[x,y+1][1]+ ((mask_coef[0,3]) * error[1]))
			imgf[x,y+1][2]= value_boundary(imgf[x,y+1][2]+ ((mask_coef[0,3]) * error[2]))

			imgf[x,y+2][0]= value_boundary(imgf[x,y+2][0]+ ((mask_coef[0,4]) * error[0]))
			imgf[x,y+2][1]= value_boundary(imgf[x,y+2][1]+ ((mask_coef[0,4]) * error[1]))
			imgf[x,y+2][2]= value_boundary(imgf[x,y+2][2]+ ((mask_coef[0,4]) * error[2]))

			imgf[x+1,y][0]= value_boundary(imgf[x+1,y][0]+ ((mask_coef[1,2]) * error[0]))
			imgf[x+1,y][1]= value_boundary(imgf[x+1,y][1]+ ((mask_coef[1,2]) * error[1]))
			imgf[x+1,y][2]= value_boundary(imgf[x+1,y][2]+ ((mask_coef[1,2]) * error[2]))

			imgf[x+2,y][0]= value_boundary(imgf[x+2,y][0]+ ((mask_coef[2,2]) * error[0]))
			imgf[x+2,y][1]= value_boundary(imgf[x+2,y][1]+ ((mask_coef[2,2]) * error[1]))
			imgf[x+2,y][2]= value_boundary(imgf[x+2,y][2]+ ((mask_coef[2,2]) * error[2]))

			imgf[x+1,y+1][0]= value_boundary(imgf[x+1,y+1][0]+ ((mask_coef[1,3]) * error[0]))
			imgf[x+1,y+1][1]= value_boundary(imgf[x+1,y+1][1]+ ((mask_coef[1,3]) * error[1]))
			imgf[x+1,y+1][2]= value_boundary(imgf[x+1,y+1][2]+ ((mask_coef[1,3]) * error[2]))

			imgf[x+1,y+2][0]= value_boundary(imgf[x+1,y+2][0]+ ((mask_coef[1,4]) * error[0]))
			imgf[x+1,y+2][1]= value_boundary(imgf[x+1,y+2][1]+ ((mask_coef[1,4]) * error[1]))
			imgf[x+1,y+2][2]= value_boundary(imgf[x+1,y+2][2]+ ((mask_coef[1,4]) * error[2]))

			imgf[x+1,y-1][0]= value_boundary(imgf[x+1,y-1][0]+ ((mask_coef[1,1]) * error[0]))
			imgf[x+1,y-1][1]= value_boundary(imgf[x+1,y-1][1]+ ((mask_coef[1,1]) * error[1]))
			imgf[x+1,y-1][2]= value_boundary(imgf[x+1,y-1][2]+ ((mask_coef[1,1]) * error[2]))

			imgf[x+1,y-2][0]= value_boundary(imgf[x+1,y-2][0]+ ((mask_coef[1,0]) * error[0]))
			imgf[x+1,y-2][1]= value_boundary(imgf[x+1,y-2][1]+ ((mask_coef[1,0]) * error[1]))
			imgf[x+1,y-2][2]= value_boundary(imgf[x+1,y-2][2]+ ((mask_coef[1,0]) * error[2]))

			imgf[x+2,y+1][0]= value_boundary(imgf[x+2,y+1][0]+ ((mask_coef[2,3]) * error[0]))
			imgf[x+2,y+1][1]= value_boundary(imgf[x+2,y+1][1]+ ((mask_coef[2,3]) * error[1]))
			imgf[x+2,y+1][2]= value_boundary(imgf[x+2,y+1][2]+ ((mask_coef[2,3]) * error[2]))

			imgf[x+2,y-1][0]= value_boundary(imgf[x+2,y-1][0]+ ((mask_coef[2,1]) * error[0]))
			imgf[x+2,y-1][1]= value_boundary(imgf[x+2,y-1][1]+ ((mask_coef[2,1]) * error[1]))
			imgf[x+2,y-1][2]= value_boundary(imgf[x+2,y-1][2]+ ((mask_coef[2,1]) * error[2]))

	return imgf

#Jarvis, Judice and Ninke Techinique
#			f(x, y)	 7/48	 5/48
#3/48  5/48   7/48 	 5/48 	 3/48
#1/48  3/48   5/48   3/48    1/48

def Jarvis_Judice_Ninke(image,m,n,activator):

	#Create an error numpy array full of zeros, for save the error that will be calculated forwards.
	error = np.zeros(3)

	#Create an zero integer matrix of shape (a,b, k), of size N times bigger than the original image
	imgf = np.zeros(shape=(m+4,n+4,3)).astype(np.uint8)
	#Copy the values of the original image to the new matrix
	copy_element(imgf,2,image)

	#Create a coeficient mask matrix
	mask_coef = np.array([[1,		1,	 	1,		7/48,	5/48],
						 [3/48,    5/48,   7/48,	5/48,	3/48],
						 [1/48,    3/48,   5/48,	3/48,	1/48]])

	#Access each point (x,y) of the matrix
	for x in range(2,m-2):
		#Calculates the order of the loop, based on which scan direction was choosed.
		lmax,lmin,flip_mask = scan_direction(x,n,2,activator)

		for y in range(lmin,lmax):
			
			#Save the old values of the point (x,y) for each channel.
			old_blue_pixel = imgf[x,y][0]
			old_green_pixel = imgf[x,y][1]
			old_red_pixel = imgf[x,y][2]
	
			
			#Generates the new pixel value - 0 or 255 (White or Black) for each channel. for each channel.
			new_red_pixel = 255 * mp.floor(old_red_pixel/128.0)
			new_green_pixel = 255 * mp.floor(old_green_pixel/128.0)
			new_blue_pixel = 255 * mp.floor(old_blue_pixel/128.0)

			#Attaches the new pixel value to the point (x,y) for each channel
			imgf[x,y][0] = new_blue_pixel
			imgf[x,y][1] = new_green_pixel
			imgf[x,y][2] = new_red_pixel
	
			#Calculates the error between the new and the old value for each channel
			error[0] = old_blue_pixel - new_blue_pixel
			error[1] = old_green_pixel - new_green_pixel 
			error[2] = old_red_pixel - new_red_pixel


			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)


			##Apply error diffusion in each channel of all the adjacent pixels
			imgf[x,y+1][0]= value_boundary(imgf[x,y+1][0]+ ((mask_coef[0,3]) * error[0]))
			imgf[x,y+1][1]= value_boundary(imgf[x,y+1][1]+ ((mask_coef[0,3]) * error[1]))
			imgf[x,y+1][2]= value_boundary(imgf[x,y+1][2]+ ((mask_coef[0,3]) * error[2]))

			imgf[x,y+2][0]= value_boundary(imgf[x,y+2][0]+ ((mask_coef[0,4]) * error[0]))
			imgf[x,y+2][1]= value_boundary(imgf[x,y+2][1]+ ((mask_coef[0,4]) * error[1]))
			imgf[x,y+2][2]= value_boundary(imgf[x,y+2][2]+ ((mask_coef[0,4]) * error[2]))

			imgf[x+1,y][0]= value_boundary(imgf[x+1,y][0]+ ((mask_coef[1,2]) * error[0]))
			imgf[x+1,y][1]= value_boundary(imgf[x+1,y][1]+ ((mask_coef[1,2]) * error[1]))
			imgf[x+1,y][2]= value_boundary(imgf[x+1,y][2]+ ((mask_coef[1,2]) * error[2]))

			imgf[x+2,y][0]= value_boundary(imgf[x+2,y][0]+ ((mask_coef[2,2]) * error[0]))
			imgf[x+2,y][1]= value_boundary(imgf[x+2,y][1]+ ((mask_coef[2,2]) * error[1]))
			imgf[x+2,y][2]= value_boundary(imgf[x+2,y][2]+ ((mask_coef[2,2]) * error[2]))

			imgf[x+1,y+1][0]= value_boundary(imgf[x+1,y+1][0]+ ((mask_coef[1,3]) * error[0]))
			imgf[x+1,y+1][1]= value_boundary(imgf[x+1,y+1][1]+ ((mask_coef[1,3]) * error[1]))
			imgf[x+1,y+1][2]= value_boundary(imgf[x+1,y+1][2]+ ((mask_coef[1,3]) * error[2]))

			imgf[x+1,y+2][0]= value_boundary(imgf[x+1,y+2][0]+ ((mask_coef[1,4]) * error[0]))
			imgf[x+1,y+2][1]= value_boundary(imgf[x+1,y+2][1]+ ((mask_coef[1,4]) * error[1]))
			imgf[x+1,y+2][2]= value_boundary(imgf[x+1,y+2][2]+ ((mask_coef[1,4]) * error[2]))

			imgf[x+1,y-1][0]= value_boundary(imgf[x+1,y-1][0]+ ((mask_coef[1,1]) * error[0]))
			imgf[x+1,y-1][1]= value_boundary(imgf[x+1,y-1][1]+ ((mask_coef[1,1]) * error[1]))
			imgf[x+1,y-1][2]= value_boundary(imgf[x+1,y-1][2]+ ((mask_coef[1,1]) * error[2]))

			imgf[x+1,y-2][0]= value_boundary(imgf[x+1,y-2][0]+ ((mask_coef[1,0]) * error[0]))
			imgf[x+1,y-2][1]= value_boundary(imgf[x+1,y-2][1]+ ((mask_coef[1,0]) * error[1]))
			imgf[x+1,y-2][2]= value_boundary(imgf[x+1,y-2][2]+ ((mask_coef[1,0]) * error[2]))

			imgf[x+2,y+1][0]= value_boundary(imgf[x+2,y+1][0]+ ((mask_coef[2,3]) * error[0]))
			imgf[x+2,y+1][1]= value_boundary(imgf[x+2,y+1][1]+ ((mask_coef[2,3]) * error[1]))
			imgf[x+2,y+1][2]= value_boundary(imgf[x+2,y+1][2]+ ((mask_coef[2,3]) * error[2]))

			imgf[x+2,y+2][2]= value_boundary(imgf[x+2,y+2][2]+ ((mask_coef[2,4]) * error[0]))
			imgf[x+2,y+2][1]= value_boundary(imgf[x+2,y+2][1]+ ((mask_coef[2,4]) * error[1]))
			imgf[x+2,y+2][2]= value_boundary(imgf[x+2,y+2][2]+ ((mask_coef[2,4]) * error[2]))

			imgf[x+2,y-1][0]= value_boundary(imgf[x+2,y-1][0]+ ((mask_coef[2,1]) * error[0]))
			imgf[x+2,y-1][1]= value_boundary(imgf[x+2,y-1][1]+ ((mask_coef[2,1]) * error[1]))
			imgf[x+2,y-1][2]= value_boundary(imgf[x+2,y-1][2]+ ((mask_coef[2,1]) * error[2]))

			imgf[x+2,y-2][0]= value_boundary(imgf[x+2,y-2][0]+ ((mask_coef[2,0]) * error[0]))
			imgf[x+2,y-2][1]= value_boundary(imgf[x+2,y-2][1]+ ((mask_coef[2,0]) * error[1]))
			imgf[x+2,y-2][2]= value_boundary(imgf[x+2,y-2][2]+ ((mask_coef[2,0]) * error[2]))


	return imgf

#Jarvis, Judice and Ninke Techinique
#							f (x, y)			32/200
#12/200 		 26/200 			   30/200			16/200 
#	    12/200				26/200				12/200
#5/200			 12/200				   12/200			5/200

def Stivenson_Arce(image,m,n,activator):

	#Create an error numpy array full of zeros, for save the error that will be calculated forwards.
	error = np.zeros(3)

	#Create an zero integer matrix of shape (a,b, k), of size N times bigger than the original image
	imgf = np.zeros(shape=(m+6,n+6,3)).astype(np.uint8)
	#Copy the values of the original image to the new matrix
	copy_element(imgf,3,image)

	#Create a coeficient mask matrix
	mask_coef = np.array([[1,			1,	 		1,		1,			1, 		32/200,		1],
						 [12/200,   	1,   	 26/200,  	1,	  	 30/200,	  1	,	  16/200],
						 [1, 	     12/200,     	1,	  26/200,		1,		 12/200,	1 ],
						 [5/200, 	   1,     	 12/200,	1,		  12/200,	   1,	  5/200 ]])

	#Access each point (x,y) of the matrix
	for x in range(3,m-3):
		#Calculates the order of the loop, based on which scan direction was choosed.
		lmax,lmin, flip_mask= scan_direction(x,n,3,activator)

		for y in range(lmin,lmax):
		
			#Save the old values of the point (x,y) for each channel.
			old_blue_pixel = imgf[x,y][0]
			old_green_pixel = imgf[x,y][1]
			old_red_pixel = imgf[x,y][2]
	
			
			#Generates the new pixel value - 0 or 255 (White or Black) for each channel. for each channel.
			new_red_pixel = 255 * mp.floor(old_red_pixel/128.0)
			new_green_pixel = 255 * mp.floor(old_green_pixel/128.0)
			new_blue_pixel = 255 * mp.floor(old_blue_pixel/128.0)

			#Attaches the new pixel value to the point (x,y) for each channel
			imgf[x,y][0] = new_blue_pixel
			imgf[x,y][1] = new_green_pixel
			imgf[x,y][2] = new_red_pixel
	
			#Calculates the error between the new and the old value for each channel
			error[0] = old_blue_pixel - new_blue_pixel
			error[1] = old_green_pixel - new_green_pixel 
			error[2] = old_red_pixel - new_red_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef= np.fliplr(mask_coef)

			#Apply error diffusion to adjacent pixels
			imgf[x,y+2][0]= value_boundary(imgf[x,y+2][0]+ ((mask_coef[1,5]) * error[0]))
			imgf[x,y+2][1]= value_boundary(imgf[x,y+2][1]+ ((mask_coef[1,5]) * error[1]))
			imgf[x,y+2][2]= value_boundary(imgf[x,y+2][2]+ ((mask_coef[1,5]) * error[2]))

			imgf[x+1,y-3][0]= value_boundary(imgf[x+1,y-3][0]+ ((mask_coef[1,0]) * error[0]))
			imgf[x+1,y-3][1]= value_boundary(imgf[x+1,y-3][1]+ ((mask_coef[1,0]) * error[1]))
			imgf[x+1,y-3][2]= value_boundary(imgf[x+1,y-3][2]+ ((mask_coef[1,0]) * error[2]))

			imgf[x+1,y-1][0]= value_boundary(imgf[x+1,y-1][0]+ ((mask_coef[1,2]) * error[0]))
			imgf[x+1,y-1][1]= value_boundary(imgf[x+1,y-1][1]+ ((mask_coef[1,2]) * error[1]))
			imgf[x+1,y-1][2]= value_boundary(imgf[x+1,y-1][2]+ ((mask_coef[1,2]) * error[2]))

			imgf[x+1,y+1][0]= value_boundary(imgf[x+1,y+1][0]+ ((mask_coef[1,4]) * error[0]))
			imgf[x+1,y+1][1]= value_boundary(imgf[x+1,y+1][1]+ ((mask_coef[1,4]) * error[1]))
			imgf[x+1,y+1][2]= value_boundary(imgf[x+1,y+1][2]+ ((mask_coef[1,4]) * error[2]))

			imgf[x+1,y+3][0]= value_boundary(imgf[x+1,y+3][0]+ ((mask_coef[1,6]) * error[0]))
			imgf[x+1,y+3][1]= value_boundary(imgf[x+1,y+3][1]+ ((mask_coef[1,6]) * error[1]))
			imgf[x+1,y+3][2]= value_boundary(imgf[x+1,y+3][2]+ ((mask_coef[1,6]) * error[2]))

			imgf[x+2,y-2][0]= value_boundary(imgf[x+2,y-2][0]+ ((mask_coef[2,1]) * error[0]))
			imgf[x+2,y-2][1]= value_boundary(imgf[x+2,y-2][1]+ ((mask_coef[2,1]) * error[1]))
			imgf[x+2,y-2][2]= value_boundary(imgf[x+2,y-2][2]+ ((mask_coef[2,1]) * error[2]))

			imgf[x+2,y][0]= value_boundary(imgf[x+2,y][0]+ ((mask_coef[2,3]) * error[0]))
			imgf[x+2,y][1]= value_boundary(imgf[x+2,y][1]+ ((mask_coef[2,3]) * error[1]))
			imgf[x+2,y][2]= value_boundary(imgf[x+2,y][2]+ ((mask_coef[2,3]) * error[2]))

			imgf[x+2,y+2][0]= value_boundary(imgf[x+2,y+2][0]+ ((mask_coef[2,5]) * error[0]))
			imgf[x+2,y+2][1]= value_boundary(imgf[x+2,y+2][1]+ ((mask_coef[2,5]) * error[1]))
			imgf[x+2,y+2][2]= value_boundary(imgf[x+2,y+2][2]+ ((mask_coef[2,5]) * error[2]))

			imgf[x+3,y-3][0]= value_boundary(imgf[x+3,y-3][0]+ ((mask_coef[3,0]) * error[0]))
			imgf[x+3,y-3][1]= value_boundary(imgf[x+3,y-3][1]+ ((mask_coef[3,0]) * error[1]))
			imgf[x+3,y-3][2]= value_boundary(imgf[x+3,y-3][2]+ ((mask_coef[3,0]) * error[2]))

			imgf[x+3,y-1][0]= value_boundary(imgf[x+3,y-1][0]+ ((mask_coef[3,2]) * error[0]))
			imgf[x+3,y-1][1]= value_boundary(imgf[x+3,y-1][1]+ ((mask_coef[3,2]) * error[1]))
			imgf[x+3,y-1][2]= value_boundary(imgf[x+3,y-1][2]+ ((mask_coef[3,2]) * error[2]))

			imgf[x+3,y+1][0]= value_boundary(imgf[x+3,y+1][0]+ ((mask_coef[3,4]) * error[0]))
			imgf[x+3,y+1][1]= value_boundary(imgf[x+3,y+1][1]+ ((mask_coef[3,4]) * error[1]))
			imgf[x+3,y+1][2]= value_boundary(imgf[x+3,y+1][2]+ ((mask_coef[3,4]) * error[2]))

			imgf[x+3,y+3][0]= value_boundary(imgf[x+3,y+3][0]+ ((mask_coef[3,6]) * error[0]))
			imgf[x+3,y+3][1]= value_boundary(imgf[x+3,y+3][1]+ ((mask_coef[3,6]) * error[1]))
			imgf[x+3,y+3][2]= value_boundary(imgf[x+3,y+3][2]+ ((mask_coef[3,6]) * error[2]))

	return imgf
