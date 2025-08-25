import cv2
import numpy as np
import os

def face_detect(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('..face_recognition/haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces,gray

def labels_for_training_data(directory):
    faces=[]
    faceId=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system file")
                continue

            id=os.path.basename()
            img_path=os.path.join(path,filename)
            print("img_path:",img_path)
            print("id:",id)
            img=cv2.imread(img_path)
            if img is None:
                print("Image is not Loaded Properly")
                continue
            faces_rect,gray = face_detect(img)
            if len(faces_rect)!= 1:
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray[y:y+w, x:x+h]
            faces.append(roi_gray)
            faceId.append(int(id))
        return  faces,faceId

def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_react(img,face):
    (x,y,w,h)=face
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

def put_text(img,text,x,y):
    cv2,put_text(img,text(x,y),cv2.FONT_HERSHEY_DUPLEX,5(255,0,0),6)







