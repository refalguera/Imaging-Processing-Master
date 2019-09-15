import cv2
import numpy as np
import math as mp

#Function that copys the elements of the original image to the center of new matrix
def copy_element(matrix,padding,copymatrix):

	matrix[padding:(matrix.shape[0] - padding),padding:(matrix.shape[1] - padding)] = copymatrix[:]
	

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

#Function that evaluets if the pixel value its on the limits, that is 0 or 255
#If the value is bigger than 255, the pixel assume the 255 value
#If is lower than 0, the pixel assume the 0 value
def value_boundary(value):
	if value > 255:
		value = 255
	if value < 0:
		value = 0

	return value


#Burkes Techinique
#		  		f(x, y) 	8/32 	4/32
#2/32  4/32 	 8/32 		4/32 	2/32

def Burkes(image,m,n,activator):

	#Create an zero integer matrix of shape (A,B), of size N times bigger than the original image
	imgf = np.zeros(shape=(m+4,n+4)).astype(np.uint8)

	#Copy the values of the original image to the new matrix
	copy_element(imgf,2,image)

	#Create a coeficient mask matrix
	mask_coef = np.array([[1,		1,	 	1,		8/32,	4/32],
						 [2/32,    4/32,   8/32,	4/32,	2/32]])

	#Access each point (x,y) of the matrix
	for x in range(2,m-2):
		#Calculate the lower and upper limets, based on the scan direction selected
		#The function also returns the flip mask activator value. If its 1, ie the ZigZag scan direction, than flip the mask. If its 0, than dont.
		lmax, lmin, flip_mask = scan_direction(x,n,2,activator)

		for y in range(lmin,lmax):
			
			#Save the old value of the point (x,y)
			old_pixel = imgf[x,y]
			# Generates the new pixel value - 0 or 255 (White or Black)
			new_pixel = 255 * mp.floor(old_pixel/128.0)
		
			#Attaches the new pixel value to the point
			imgf[x,y] = new_pixel

			#Calculates the error between the new and the old value
			error = old_pixel - new_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)

			#Apply error diffusion to adjacent pixels
			imgf[x+1,y] = value_boundary(imgf[x+1,y] + ((mask_coef[1,2]) * error))
			imgf[x,y+1] = value_boundary(imgf[x,y+1] + ((mask_coef[0,3]) * error))
			imgf[x+1,y+1] = value_boundary(imgf[x+1,y+1] + ((mask_coef[1,3]) * error))
			imgf[x,y+2] = value_boundary(imgf[x,y+2] + ((mask_coef[0,4]) * error))
			imgf[x+1,y+2] = value_boundary(imgf[x+1,y+2]  + ((mask_coef[1,4]) * error))
			imgf[x+1,y-1] = value_boundary(imgf[x+1,y-1] + ((mask_coef[1,1]) * error))
			imgf[x+1,y-2] = value_boundary(imgf[x+1,y-2] + ((mask_coef[1,0]) * error))


	return imgf

#Floyd and Steinberg Techinique
#		 f(x, y) 7/16
#	3/16  5/16   1/16

def Floyd_Stein(image,m,n,activator):

	#Create an zero integer matrix of shape (A,B), of size N times bigger than the original image
	imgf = np.zeros(shape=(m+2,n+2)).astype(np.uint8)

	#Copy the values of the original image to the new matrix
	copy_element(imgf,1,image)


	#Create a coeficient mask matrix
	mask_coef = np.array([[1,	1,	7/16],
						  [3/16,5/16,1/16]])

	#Acess all the points (x,y) of the matrix
	for x in range(1,m-1):
		#Calculate the lower and upper limets, based on the scan direction selected
		#The function also returns the flip mask activator value. If its 1, ie the ZigZag scan direction, than flip the mask. If its 0, than dont.
		lmax, lmin, flip_mask = scan_direction(x,n,1,activator)
		for y in range(lmin,lmax):
			#Save the old value of the point (x,y)
			old_pixel = imgf[x,y]

			# Generates the new pixel value - 0 or 255 (White or Black)
			new_pixel = 255 * mp.floor(old_pixel/128.0)

			#Attaches the new pixel value to the point
			imgf[x,y] = new_pixel

			#Calculates the error between the new and the old value
			error = old_pixel - new_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)

			#Apply error diffusion to adjacent pixels
			imgf[x+1,y] = value_boundary(imgf[x+1,y] + ((mask_coef[1,1]) * error))
			imgf[x,y+1] = value_boundary(imgf[x,y+1] + ((mask_coef[0,2]) * error))
			imgf[x+1,y-1] = value_boundary(imgf[x+1,y-1] + ((mask_coef[1,0]) * error))
			imgf[x+1,y+1] = value_boundary(imgf[x+1,y+1]  + ((mask_coef[1,2]) * error))
	
	return imgf


#Stucki Techinique
#			f(x, y)	 8/42 	4/42
#2/42  4/42   8/42 	 4/42 	2/42
#1/42  2/42   4/42   2/42    1/4

def Stuki(image,m,n,activator):

	#Create an zero integer matrix of shape (A,B), of size N times bigger than the original image
	imgf = np.zeros(shape=(m+4,n+4)).astype(np.uint8)
	#Copy the values of the original image to the new matrix
	copy_element(imgf,2,image)

	#Create a coeficient mask matrix
	mask_coef = np.array([[1,		1,	 	1,		8/42,	4/42],
						 [2/42,    4/42,   8/42,	4/42,	2/42],
						 [1/42,    2/42,   4/42,	2/42,	1/42]])

	#Acess all the points (x,y) of the matrix
	for x in range(2,m-2):
		#Calculate the lower and upper limets, based on the scan direction selected
		#The function also returns the flip mask activator value. If its 1, ie the ZigZag scan direction, than flip the mask. If its 0, than dont.
		lmax,lmin,flip_mask = scan_direction(x,n,2,activator)

		for y in range(lmin,lmax):
			#Save the old value of the point (x,y)
			old_pixel = imgf[x,y]
			# Generates the new pixel value - 0 or 255 (White or Black)
			new_pixel = 255 * mp.floor(old_pixel/128.0)
			
			#Attaches the new pixel value to the point
			imgf[x,y] = new_pixel

			#Calculates the error between the new and the old value
			error = old_pixel - new_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)

			#Apply error diffusion to adjacent pixels
			imgf[x,y+1] = value_boundary(imgf[x,y+1] + ((mask_coef[0,3]) * error))
			imgf[x,y+2] = value_boundary(imgf[x,y+2] + ((mask_coef[0,4]) * error))
			imgf[x+1,y] = value_boundary(imgf[x+1,y] + ((mask_coef[1,2]) * error))
			imgf[x+2,y] = value_boundary(imgf[x+2,y] + ((mask_coef[2,2]) * error))
			imgf[x+1,y+1] = value_boundary(imgf[x+1,y+1] + ((mask_coef[1,3]) * error))
			imgf[x+1,y+2] = value_boundary(imgf[x+1,y+2] + ((mask_coef[1,4]) * error))
			imgf[x+1,y-1] = value_boundary(imgf[x+1,y-1] + ((mask_coef[1,1]) * error))
			imgf[x+1,y-2] = value_boundary(imgf[x+1,y-2] + ((mask_coef[1,0]) * error))
			imgf[x+2,y+1] = value_boundary(imgf[x+2,y+1] + ((mask_coef[2,3]) * error))
			imgf[x+2,y+2] = value_boundary(imgf[x+2,y+2] + ((mask_coef[2,4]) * error))
			imgf[x+2,y-1] = value_boundary(imgf[x+2,y-1] + ((mask_coef[2,1]) * error))
			imgf[x+2,y-2] = value_boundary(imgf[x+2,y-2] + ((mask_coef[2,0]) * error))


	return imgf

#Sierra Techinique
#			f(x, y)	 5/32 	3/32
#2/32 4/32   5/32 	 4/32  	2/32
#	  2/32   3/32    2/32

def Sierra(image,m,n,activator):

	imgf = np.zeros(shape=(m+4,n+4)).astype(np.uint8)
	copy_element(imgf,2,image)

	mask_coef = np.array([[1,		1,	 	1,		5/32,	3/32],
						 [2/32,    4/32,   5/32,	4/32,	2/32],
						 [1, 	   2/32,   3/32,	2/32,	 1]])

	#Acess all the points (x,y) of the matrix
	for x in range(2,m-2):
		#Calculate the lower and upper limets, based on the scan direction selected
		#The function also returns the flip mask activator value. If its 1, ie the ZigZag scan direction, than flip the mask. If its 0, than dont.
		lmax,lmin, flip_mask = scan_direction(x,n,2,activator)

		for y in range(lmin,lmax):
			#Save the old value of the point (x,y)
			old_pixel = imgf[x,y]
			# Generates the new pixel value - 0 or 255 (White or Black)
			new_pixel = 255 * mp.floor(old_pixel/128.0)
		
			#Attaches the new pixel value to the point
			imgf[x,y] = new_pixel

			#Calculates the error between the new and the old value
			error = old_pixel - new_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)

			#Apply error diffusion to adjacent pixels
			imgf[x,y+1] = value_boundary(imgf[x,y+1] + ((mask_coef[0,3]) * error))
			imgf[x,y+2] = value_boundary(imgf[x,y+2] + ((mask_coef[0,4]) * error))
			imgf[x+1,y] = value_boundary(imgf[x+1,y] + ((mask_coef[1,2]) * error))
			imgf[x+2,y] = value_boundary(imgf[x+2,y] + ((mask_coef[2,2]) * error))
			imgf[x+1,y+1] = value_boundary(imgf[x+1,y+1] + ((mask_coef[1,3]) * error))
			imgf[x+1,y+2] = value_boundary(imgf[x+1,y+2] + ((mask_coef[1,4]) * error))
			imgf[x+1,y-1] = value_boundary(imgf[x+1,y-1] + ((mask_coef[1,1]) * error))
			imgf[x+1,y-2] = value_boundary(imgf[x+1,y-2] + ((mask_coef[1,0]) * error))
			imgf[x+2,y+1] = value_boundary(imgf[x+2,y+1] + ((mask_coef[2,3]) * error))
			imgf[x+2,y-1] = value_boundary(imgf[x+2,y-1] + ((mask_coef[2,1]) * error))
			

	return imgf


#Jarvis, Judice and Ninke Techinique
#			f(x, y)	 7/48	 5/48
#3/48  5/48   7/48 	 5/48 	 3/48
#1/48  3/48   5/48   3/48    1/48

def Jarvis_Judice_Ninke(image,m,n,activator):

	imgf = np.zeros(shape=(m+4,n+4)).astype(np.uint8)
	copy_element(imgf,2,image)

	mask_coef = np.array([[1,		1,	 	1,		7/48,	5/48],
						 [3/48,    5/48,   7/48,	5/48,	3/48],
						 [1/48,    3/48,   5/48,	3/48,	1/48]])

	#Acess all the points (x,y) of the matrix
	for x in range(2,m-2):
		lmax,lmin,flip_mask = scan_direction(x,n,2,activator)

		for y in range(lmin,lmax):
			#Save the old value of the point (x,y)
			old_pixel = imgf[x,y]
			# Generates the new pixel value - 0 or 255 (White or Black)
			new_pixel = 255 * mp.floor(old_pixel/128.0)
		
			#Attaches the new pixel value to the point
			imgf[x,y] = new_pixel

			#Calculates the error between the new and the old value
			error = old_pixel - new_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)

			#Apply the Error Difusion 
			imgf[x,y+1] = value_boundary(imgf[x,y+1] + ((mask_coef[0,3]) * error))
			imgf[x,y+2] = value_boundary(imgf[x,y+2] + ((mask_coef[0,4]) * error))
			imgf[x+1,y] = value_boundary(imgf[x+1,y] + ((mask_coef[1,2]) * error))
			imgf[x+2,y] = value_boundary(imgf[x+2,y] + ((mask_coef[2,2]) * error))
			imgf[x+1,y+1] = value_boundary(imgf[x+1,y+1] + ((mask_coef[1,3]) * error))
			imgf[x+1,y+2] = value_boundary(imgf[x+1,y+2] + ((mask_coef[1,4]) * error))
			imgf[x+1,y-1] = value_boundary(imgf[x+1,y-1] + ((mask_coef[1,1]) * error))
			imgf[x+1,y-2] = value_boundary(imgf[x+1,y-2] + ((mask_coef[1,0]) * error))
			imgf[x+2,y+1] = value_boundary(imgf[x+2,y+1] + ((mask_coef[2,3]) * error))
			imgf[x+2,y+2] = value_boundary(imgf[x+2,y+2] + ((mask_coef[2,4]) * error))
			imgf[x+2,y-1] = value_boundary(imgf[x+2,y-1] + ((mask_coef[2,1]) * error))
			imgf[x+2,y-2] = value_boundary(imgf[x+2,y-2] + ((mask_coef[2,0]) * error))


	return imgf

#Jarvis, Judice and Ninke Techinique
#							f (x, y)			32/200
#12/200 		 26/200 			   30/200			16/200 
#	    12/200				26/200				12/200
#5/200			 12/200				   12/200			5/200

def Stivenson_Arce(image,m,n,activator):

	imgf = np.zeros(shape=(m+6,n+6)).astype(np.uint8)
	copy_element(imgf,3,image)

	mask_coef = np.array([[1,			1,	 		1,		1,			1, 		32/200,		1],
						 [12/200,   	1,   	 26/200,  	1,	  	 30/200,	  1	,	  16/200],
						 [1, 	     12/200,     	1,	  26/200,		1,		 12/200,	1 ],
						 [5/200, 	   1,     	 12/200,	1,		  12/200,	   1,	  5/200 ]])

	#Acess all the points (x,y) of the matrix
	for x in range(3,m-3):
		lmax, lmin, flip_mask = scan_direction(x,n,3,activator)
		for y in range(lmin,lmax):
			#Save the old value of the point (x,y)
			old_pixel = imgf[x,y]
			# Generates the new pixel value - 0 or 255 (White or Black)
			new_pixel = 255 * mp.floor(old_pixel/128.0)

			#Attaches the new pixel value to the point
			imgf[x,y] = new_pixel

			#Calculates the error between the new and the old value
			error = old_pixel - new_pixel

			#Flip each line of the coefficient mask matrix.
			if(flip_mask == 1):
				mask_coef = np.fliplr(mask_coef)

			#Apply error diffusion to adjacent pixels
			imgf[x,y+2] = value_boundary(imgf[x,y+2] + ((mask_coef[1,5]) * error))
			imgf[x+1,y-3] = value_boundary(imgf[x+1,y-3] + ((mask_coef[1,0]) * error))
			imgf[x+1,y-1] = value_boundary(imgf[x+1,y-1] + ((mask_coef[1,2]) * error))
			imgf[x+1,y+1] = value_boundary(imgf[x+1,y+1] + ((mask_coef[1,4]) * error))
			imgf[x+1,y+3] = value_boundary(imgf[x+1,y+3] + ((mask_coef[1,6]) * error))
			imgf[x+2,y-2] = value_boundary(imgf[x+2,y-2] + ((mask_coef[2,1]) * error))
			imgf[x+2,y] = value_boundary(imgf[x+2,y] + ((mask_coef[2,3]) * error))
			imgf[x+2,y+2] = value_boundary(imgf[x+2,y+2] + ((mask_coef[2,5]) * error))
			imgf[x+3,y-3] = value_boundary(imgf[x+3,y-3] + ((mask_coef[3,0]) * error))
			imgf[x+3,y-1] = value_boundary(imgf[x+3,y-1] + ((mask_coef[3,2]) * error))
			imgf[x+3,y+1] = value_boundary(imgf[x+3,y+1] + ((mask_coef[3,4]) * error))
			imgf[x+3,y+3] = value_boundary(imgf[x+3,y+3] + ((mask_coef[3,6]) * error))


	return imgf
