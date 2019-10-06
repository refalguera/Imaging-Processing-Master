import cv2
import argparse 
import objectmeasures as measures

def main():


	parser = argparse.ArgumentParser()

	#Input
	parser.add_argument('-img',
						'--img_address',
						required = True,
						help='Selected Image')


	args = parser.parse_args()

	filename_input = args.img_address

	#Open the RGB images in grayscale using opencv. The function imread immediately save the image in numpy array format.
	imageColor = cv2.imread(filename_input) 
	
	#Save results
	cv2.imwrite('output/Color_Transformation/ColorTransformation' + filename_input,measures.Color_Transformation(imageColor))
	cv2.imwrite('output/Object_Contours/ContoursExtraction' + filename_input,measures.Object_Contour_Extraction(imageColor))
	cv2.imwrite('output/Feature_Extraction/Object_Feature' + filename_input,measures.Object_Features_Extraction(imageColor,filename_input))
	
	
if __name__ == '__main__':
	main()
