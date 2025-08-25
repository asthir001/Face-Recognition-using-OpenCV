#import required libraries
import numpy as np
import cv2

#define our classifiers
#load the required xml files

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade=cv2.CascadeClassifier('haarcascades_eye.xml')

#read the image

img=cv2.imread('IMG-20191001-WA0115.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect multiscale objects
faces=face_cascade.detectMultiScale(gray, 1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray=gray[y:y+h, x:x+w]
    roi_color=img[y:y+h, x:x+w]

    cv2.imshow('Grey',roi_gray)

    eyes= eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(255,0,0),2)

cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()