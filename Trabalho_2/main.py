import cv2
import argparse 
import globalmethod as global_method
import bernsenmethod as bernsen_method
import niblackmethod as niblack_method
import sauvola_pietaksinen_method as sv_method
import phansalskar_more_sabale_method as pms_method
import contrastmethod as contrast_method
import averagemethod as average_method
import medianmethod as median_method

def main():


	parser = argparse.ArgumentParser()

	#Input
	parser.add_argument('-img',
						'--img_address',
						required = True,
						help='Selected Image')

	parser.add_argument('-hist',
						'--hist_address',
						required = True,
						help='Selected Image')

	args = parser.parse_args()

	filename_input = args.img_address
	hist_name = args.hist_address

	#Open the RGB images in grayscale using opencv. The function imread immediately save the image in numpy array format.
	imageMono = cv2.imread(filename_input,-1) 

	#Save the Monochromatic images after the thresholding 
	cv2.imwrite("output/Global/Global_Method_" + filename_input ,global_method.Global_Thresholding(imageMono,hist_name))
	cv2.imwrite("output/Bernsen/Bernsen_Method_" + filename_input ,bernsen_method.Bernsen_Thresholding(imageMono,hist_name))
	cv2.imwrite("output/Niblack/Niblack_Method_" + filename_input ,niblack_method.Niblack_Thresholding(imageMono,hist_name))
	cv2.imwrite("output/SauvolaPietaksinen/Sauvola_Pietaksinen_Method_" + filename_input,sv_method.Sauvola_Pietaksinen_Thresholding(imageMono,hist_name))
	cv2.imwrite("output/PhansalskarMoreSabale/Phansalskar_More_Sabale_Method_" + filename_input,pms_method.Phansalskar_More_Sabale_Thresholding(imageMono,hist_name))
	cv2.imwrite("output/Contrast/Contrast_Method_" + filename_input ,contrast_method.Contrast_Thresholding(imageMono,hist_name))
	cv2.imwrite("output/Average/Average_Method_" + filename_input,average_method.Average_Thresholding(imageMono,hist_name))
	cv2.imwrite("output/Median/Median_Method_" + filename_input,median_method.Median_Thresholding(imageMono,hist_name))
	
if __name__ == '__main__':
	main()

