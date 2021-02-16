import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import struct
import time

from PIL import Image

#=====================================================================
#Sample code no camera
#=====================================================================

im = Image.open('sample3.bmp')
im.save('sample3.png')
font = cv.FONT_HERSHEY_COMPLEX

img = cv.imread('sample3.png',0)
ret,thresh1 = cv.threshold(img,210,255,cv.THRESH_BINARY) #used for the Contour map
#These are other thresholding options we should only need thresh1  leaving these here for now
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,210,255,cv.THRESH_TOZERO) 
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
#prints allth threshold options and original grayscale image
plt.show()

#=====================================================================
#Find Contours
#=====================================================================

# Detecting contours in image. 
contours, _= cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
  
# Going through every contours found in the image. 
for cnt in contours : 
  
	approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True) 
  
    # draws boundary of contours. 
	area = cv.contourArea(approx)

	#Only draws contours for large areas, to help prevent taking wrong temperature
	if area > 50000:
		cv.drawContours(img, [approx], 0, (0, 0, 255), 5)

		mask = np.zeros(thresh1.shape,np.uint8)
		cv.drawContours(mask,[cnt],0,255,-1)

		#array of all pixels in mask(Dont think we need this if we can take mean)
		pixelpoints = cv.findNonZero(mask)

		#Mean grayscale values of the mask
		mean_val = cv.mean(img,mask = mask)
  
		# Used to flatted the array containing 
		# the co-ordinates of the vertices. 
		#this is just labeling the contour lines  really just using this for a
		#screenshot for the presentation
		n = approx.ravel()
		i = 0
  
		for j in n :
			if(i % 2 == 0): 
				x = n[i]
				y = n[i + 1]
  
				# String containing the co-ordinates. 
				string = str(x) + " " + str(y)  
  
				if(i == 0): 
					# text on topmost co-ordinate. 
					cv.putText(img, "Arrow tip", (x, y), font, 0.5, (255, 0, 0))  
				else: 
					# text on remaining co-ordinates. 
					cv.putText(img, string, (x, y), font, 0.5, (0, 255, 0))  
			i = i + 1
  
# Showing the Mask image. 
cv.imshow('Mask', thresh1)
# Exiting the window if 'q' is pressed on the keyboard. 
if cv.waitKey(0) & 0xFF == ord('q'):  
    cv.destroyAllWindows()  

# Showing the Final image.
cv.imshow('ROI', img)
  
# Exiting the window if 'q' is pressed on the keyboard. 
if cv.waitKey(0) & 0xFF == ord('q'):  
    cv.destroyAllWindows()


print(mean_val)

exit(0)

#=====================================================================

#=====================================================================
#Code outline for camera
#=====================================================================

camera_num = 0
 
for camera_num in range(6):
	cam = cv.VideoCapture(camera_num)
	if not cam.isOpened():
		print("Was not able to open camera", camera_num)
		cam.release()
		continue
	if not cam.set(3, 240):
		print("Was not able to set camera", camera_num, "width to 240 pixels")
		cam.release()
		continue
	if not cam.set(4, 321):
		print("Was not able to set camera", camera_num, "height to 321 pixels")
		cam.release()
		continue
	if cam.get(3) != 240:
		print("Was not able to set camera", camera_num, "width to 240 pixels")
		cam.release()
		continue
	break
 
print("Camera %d open at size: (%d x %d) %d FPS" % (camera_num, cam.get(3), cam.get(4), cam.get(5)))
 
cv.namedWindow('Thermal Camera - Press Q to quit', cv.WINDOW_NORMAL)
cv.resizeWindow('Thermal Camera - Press Q to quit', 480, 642)
 
while(True):
	# Capture frame-by-frame
	ret, frame = cam.read()
 
	if not ret:
		print("Failed to fetch frame")
		time.sleep(0.1)
		continue
	#print("Frame OK!")
#=====================================================================
    #Our Code here i believe
#=====================================================================
	#colorframe = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
	gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	ret,thresh1 = cv.threshold(gray_frame,210,255,cv.THRESH_BINARY) #used for the Contour map

#=====================================================================
#Find Contours
#=====================================================================

	# Detecting contours in image. 
	contours, _= cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
  
	# Going through every contours found in the image. 
	for cnt in contours : 
  
		approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True) 
  
    	# draws boundary of contours. 
		area = cv.contourArea(approx)

		#Only draws contours for large areas, to help prevent taking wrong temperature
		if area > 50000:
			cv.drawContours(gray_frame, [approx], 0, (0, 0, 255), 5)

			mask = np.zeros(thresh1.shape,np.uint8)
			cv.drawContours(mask,[cnt],0,255,-1)

			#array of all pixels in mask(Dont think we need this if we can take mean)
			pixelpoints = cv.findNonZero(mask)

			#Mean grayscale values of the mask
			mean_val = cv.mean(gray_frame,mask = mask)
  
			# Used to flatted the array containing 
			# the co-ordinates of the vertices. 
			#this is just labeling the contour lines  really just using this for a
			#screenshot for the presentation
			n = approx.ravel()
			i = 0
  
			for j in n :
				if(i % 2 == 0): 
					x = n[i]
					y = n[i + 1]
  
					# String containing the co-ordinates. 
					string = str(x) + " " + str(y)  
  
					if(i == 0): 
						# text on topmost co-ordinate. 
						cv.putText(gray_frame, "Arrow tip", (x, y), font, 0.5, (255, 0, 0))  
					else: 
						# text on remaining co-ordinates. 
						cv.putText(gray_frame, string, (x, y), font, 0.5, (0, 255, 0))  
				i = i + 1
  
	# Showing the Mask image. 
	cv.imshow('Mask', thresh1)
	# Exiting the window if 'q' is pressed on the keyboard. 
	if cv.waitKey(0) & 0xFF == ord('q'):  
		cv.destroyAllWindows()  

	# Showing the Final image.
	cv.imshow('ROI', gray_frame)
  
	# Exiting the window if 'q' is pressed on the keyboard. 
	if cv.waitKey(0) & 0xFF == ord('q'):  
		cv.destroyAllWindows()


	print(mean_val)
#=====================================================================

	# Display the resulting frame
	cv.imshow('Thermal Camera - Press Q to quit', gray_frame)
 
	cam.set(cv.CAP_PROP_CONVERT_RGB, 0)
	ret, frame = cam.read()
	if not ret:
		print("Failed to fetch frame")
		time.sleep(0.1)
		continue
	print("Temp calculation (experimental): ", end="" )
	print(struct.unpack("h", frame[320][0][0:2])[0]/10)
	cam.set(cv.CAP_PROP_CONVERT_RGB, 1)
#=====================================================================
    
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
 
	#yuvframe = cv.cvtColor(frame, cv.COLOR_RGB2YUV)
	#print(yuvframe[-1][0:3])

cam.release()
cv.destroyAllWindows()

#=====================================================================