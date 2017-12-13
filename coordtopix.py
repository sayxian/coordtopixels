import cv2
import numpy as np
import pyautogui
import os
import sys
import msvcrt
import time

"""
Super buggy coord to pix program that takes the mean of the 4 pixels and
average it out.

1. Run the program, and quickly switch over to area of capture.
2. Drag mouse over area of interest ( rectangle of interest)
3. Mean is displayed after selecting the area and pressing k to display.

"""

##outwards=msvcrt.getch()
##while outwards == " ":
time.sleep(2.5)
##    outwards=msvcrt.getch()
##    print('please press c')

image = pyautogui.screenshot('image.png')
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False
getROI = False
refPt = []

# load the image, clone it, and setup the mouse callback function
##image = cv2.imread(args["image"])
image = cv2.imread('image.png')
clone = image.copy()


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, getROI

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        x_end, y_end = x, y
        cropping = False
        getROI = True

 
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
font = cv2.FONT_HERSHEY_SIMPLEX
meanIntensity=0
# keep looping until the 'q' key is pressed
while True:
    i = image.copy()
    if not cropping and not getROI:
        cv2.imshow("image", image)

    elif cropping and not getROI:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("image", i)

    elif not cropping and getROI:
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("image", image)


    key = cv2.waitKey(1) & 0xFF
 
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
        getROI = False
 
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        cv2.destroyAllWindows()
        os.execl(sys.executable, sys.executable, *sys.argv)

    elif key == ord("z"):
        break

    if key == ord("k"):
        x1=x_end-x_start
        y1=y_end-y_start
        mask = np.zeros((x1,y1),np.uint8)
        mean_val = cv2.mean(image[x_start:x_end,y_start:y_end])    
        meanIntensity1=(sum(mean_val)/(len(mean_val)))
        if meanIntensity1 !=0 and abs(meanIntensity1 - meanIntensity)>1:
            print(meanIntensity1)
            cv2.putText(image,str(meanIntensity1),(10,500), font, 4,(0,0,0),2,cv2.LINE_AA)
            text_file = open("IntensityOutput.txt", "a")
            text_file.write("\n Intensity: %f " %meanIntensity1)
            text_file.close()
        meanIntensity = meanIntensity1
        
 
# if there are two reference points, then crop the region of interest
# from the image and display it
refPt = [(x_start, y_start), (x_end, y_end)]
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)

    hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    print('min H = {}, min S = {}, min V = {}; max H = {}, max S = {}, max V = {}'.format(hsvRoi[:,:,0].min(), hsvRoi[:,:,1].min(), hsvRoi[:,:,2].min(), hsvRoi[:,:,0].max(), hsvRoi[:,:,1].max(), hsvRoi[:,:,2].max()))
 
    lower = np.array([hsvRoi[:,:,0].min(), hsvRoi[:,:,1].min(), hsvRoi[:,:,2].min()])
    upper = np.array([hsvRoi[:,:,0].max(), hsvRoi[:,:,1].max(), hsvRoi[:,:,2].max()])

    image_to_thresh = clone
    hsv = cv2.cvtColor(image_to_thresh, cv2.COLOR_BGR2HSV)

    kernel = np.ones((3,3),np.uint8)
    # for red color we need to masks.
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)


