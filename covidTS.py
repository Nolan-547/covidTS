import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import struct
import time

from PIL import Image

font = cv.FONT_HERSHEY_COMPLEX

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
flag = True
while(flag):
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
	#color_frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
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
		if area > 800:
			cv.drawContours(gray_frame, [approx], 0, (0, 0, 255), 5)

			mask = np.zeros(thresh1.shape,np.uint8)
			cv.drawContours(mask,[cnt],0,255,-1)

			#array of all pixels in mask(Dont think we need this if we can take mean)
			pixelpoints = cv.findNonZero(mask)

			#Mean grayscale values of the mask
			mean_val = cv.mean(frame,mask = mask)
			topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
			#mean_val = cv.mean(color_frame,mask = mask)
  
	# Showing the Mask image. 
	cv.imshow('Mask', thresh1)
	# Exiting the window if 'q' is pressed on the keyboard. 
	if cv.waitKey(1) & 0xFF == ord('q'):  
		cv.destroyAllWindows()  

	# Showing the Final image.
	cv.imshow('ROI', gray_frame)
  
	# Exiting the window if 'q' is pressed on the keyboard. 
	if cv.waitKey(1) & 0xFF == ord('q'):  
		cv.destroyAllWindows()


	print(mean_val)
	#print(gray_frame[topmost[0]][topmost[1]])
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
	print(struct.unpack("h", frame[320][0][0:2])[0])
	cam.set(cv.CAP_PROP_CONVERT_RGB, 1)
#=====================================================================
    
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
 
	#yuvframe = cv.cvtColor(frame, cv.COLOR_RGB2YUV)
	#print(yuvframe[-1][0:3])
	#flag = False

cam.release()
cv.destroyAllWindows()

#=====================================================================