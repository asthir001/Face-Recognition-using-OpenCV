#import required libraries


import numpy as np
import cv2 
import imutils

#define our classifiers
#load the required xml files

hog=cv2.HOGDESCRIPTOR()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap= cv2.VideoCapture('location of the video')

while cap.isOpened():
    ret, image=cap.read()
    if ret:
        image = imutils.resize(image,width=min(400,image.shape[1]))
        for (x,y,w,h) in faces:
            cv2.rectangle(img,
                        (x,y),
                        (x+w,y+h),
                        (0,255,0),
                        2)
            roi_gray=gray[y:y+h, x:x+w]
            roi_color=img[y:y+h, x:x+w]

            cv2.imshow('Grey',roi_gray)

            eyes= eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,eq,eh) in eyes:
                cv2.rectangle(roi_color,
                            (ex,ey),
                            (ex+ew, ey+eh),
                            (0,255,0),
                            2)

        cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()





