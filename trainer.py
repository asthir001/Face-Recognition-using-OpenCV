import cv2
import os
import numpy as np
import face_Recognition as fr


    # Read the input image
img = cv2.imread('../face_recognition/27.jpg')
face_cascade,gray = fr.face_detect(img)
print("Face Detected:",face_cascade)



faces,faceId=fr.labels_for_training_data('../face_recognition/photo')
face_recognizer=fr.train_classifier(faces,faceId)
name={00:"Person_1",11:"Person_2"}

for face in face_cascade:
    (x,y,w,h,)=face
    roi_gray = gray[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(img,face)
    predicted_name=name[label]
    fr.put_text(img,predicted_name,x,y)

resized_img=cv2.resize(img,(1000,700))
cv2.imshow("img",resized_img)
cv2.waitKey()
cv2.destroyAllWindows()




