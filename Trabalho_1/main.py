import cv2
import argparse
import monochromatic as mono
import RGB as colorscript


def main():
	
	parser = argparse.ArgumentParser()

	#Input
	parser.add_argument('-img',
						'--img_adress',
						required = True,
						help='Selected Image')

	args = parser.parse_args()

	filename_input = args.img_adress

	#Open the RGB images in grayscale using opencv. The function imread immediately save the image in numpy array format.
	imageMono = cv2.imread(filename_input,0) 

	#Open the RGB images using opencv. The function imread immediately save the image in numpy array format.
	imageRGB = cv2.imread(filename_input) 


	# Save final image - Monocromatic
	cv2.imwrite("output/Burkes/Monochromatic/Normal_Order-" + filename_input, mono.Burkes(imageMono,imageMono.shape[0],imageMono.shape[1],0))
	cv2.imwrite("output/Burkes/Monochromatic/ZigZag_Order-" + filename_input,mono.Burkes(imageMono,imageMono.shape[0],imageMono.shape[1],1))
	cv2.imwrite("output/Floyd_Steinberg/Monochromatic/Normal_Order-" + filename_input,mono.Floyd_Stein(imageMono,imageMono.shape[0],imageMono.shape[1],0))
	cv2.imwrite("output/Floyd_Steinberg/Monochromatic/ZigZag_Order-" + filename_input,mono.Floyd_Stein(imageMono,imageMono.shape[0],imageMono.shape[1],1))
	cv2.imwrite("output/Stuki/Monochromatic/Normal_Order-" + filename_input,mono.Stuki(imageMono,imageMono.shape[0],imageMono.shape[1],0))
	cv2.imwrite("output/Stuki/Monochromatic/ZigZag_Order-" + filename_input,mono.Stuki(imageMono,imageMono.shape[0],imageMono.shape[1],1))
	cv2.imwrite("output/Sierra/Monochromatic/Normal_Order-" + filename_input,mono.Sierra(imageMono,imageMono.shape[0],imageMono.shape[1],0))
	cv2.imwrite("output/Sierra/Monochromatic/ZigZag_Order-" + filename_input,mono.Sierra(imageMono,imageMono.shape[0],imageMono.shape[1],1))
	cv2.imwrite("output/Jarvis_Judice_Ninke/Monochromatic/Normal_Order-" + filename_input,mono.Jarvis_Judice_Ninke(imageMono,imageMono.shape[0],imageMono.shape[1],0))
	cv2.imwrite("output/Jarvis_Judice_Ninke/Monochromatic/ZigZag_Order-" + filename_input,mono.Jarvis_Judice_Ninke(imageMono,imageMono.shape[0],imageMono.shape[1],1))
	cv2.imwrite("output/Stivenson_Arce/Monochromatic/Normal_Order-" + filename_input,mono.Stivenson_Arce(imageMono,imageMono.shape[0],imageMono.shape[1],0))
	cv2.imwrite("output/Stivenson_Arce/Monochromatic/ZigZag_Order-" + filename_input,mono.Stivenson_Arce(imageMono,imageMono.shape[0],imageMono.shape[1],1))

	# Save final image - RGB
	cv2.imwrite('output/Burkes/RGB/Normal_Order-' + filename_input, colorscript.Burkes(imageRGB,imageRGB.shape[0],imageRGB.shape[1],0))
	cv2.imwrite("output/Burkes/RGB/ZigZag_Order-" + filename_input,colorscript.Burkes(imageRGB,imageRGB.shape[0],imageRGB.shape[1],1))
	cv2.imwrite("output/Floyd_Steinberg/RGB/Normal_Order-" + filename_input,colorscript.Floyd_Stein(imageRGB,imageRGB.shape[0],imageRGB.shape[1],0))
	cv2.imwrite("output/Floyd_Steinberg/RGB/ZigZag_Order-" + filename_input,colorscript.Floyd_Stein(imageRGB,imageRGB.shape[0],imageRGB.shape[1],1))
	cv2.imwrite("output/Stuki/RGB/Normal_Order-" + filename_input,colorscript.Stuki(imageRGB,imageRGB.shape[0],imageRGB.shape[1],0))
	cv2.imwrite("output/Stuki/RGB/ZigZag_Order-" + filename_input,colorscript.Stuki(imageRGB,imageRGB.shape[0],imageRGB.shape[1],1))
	cv2.imwrite("output/Sierra/RGB/Normal_Order-" + filename_input,colorscript.Sierra(imageRGB,imageRGB.shape[0],imageRGB.shape[1],0))
	cv2.imwrite("output/Sierra/RGB/ZigZag_Order-" + filename_input,colorscript.Sierra(imageRGB,imageRGB.shape[0],imageRGB.shape[1],1))
	cv2.imwrite("output/Jarvis_Judice_Ninke/RGB/Normal_Order-" + filename_input,colorscript.Jarvis_Judice_Ninke(imageRGB,imageRGB.shape[0],imageRGB.shape[1],0))
	cv2.imwrite("output/Jarvis_Judice_Ninke/RGB/ZigZag_Order-" + filename_input,colorscript.Jarvis_Judice_Ninke(imageRGB,imageRGB.shape[0],imageRGB.shape[1],1))
	cv2.imwrite("output/Stivenson_Arce/RGB/Normal_Order-" + filename_input,colorscript.Stivenson_Arce(imageRGB,imageRGB.shape[0],imageRGB.shape[1],0))
	cv2.imwrite("output/Stivenson_Arce/RGB/ZigZag_Order-" + filename_input,colorscript.Stivenson_Arce(imageRGB,imageRGB.shape[0],imageRGB.shape[1],1))

if __name__ == '__main__':
	main()