import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")


cap = cv2.VideoCapture(0)

i = 0

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        modifier = 10
        x_start = x-modifier
        x_end = x+w+modifier
        y_start = y-modifier
        y_end = y+h+modifier

        roi_gray = gray[y_start:y_end, x_start:x_end]
        
        print(x_start, y_start, x_start, y_start)

        if i % 10 == 0:
            img_item = "my_image"+str(i)+".png"
            cv2.imwrite(img_item, roi_gray)
            print("image saved")
        
        i += 1

        color = (255,0,0)
        stroke = 2
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, stroke)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()