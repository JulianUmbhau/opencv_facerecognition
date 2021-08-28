import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        modifier = 0
        x_start = x-modifier
        x_end = x+w+modifier
        y_start = y-modifier
        y_end = y+h+modifier

        roi_gray = gray[y_start:y_end, x_start:x_end]
        
        print(x_start, y_start, x_start, y_start)

        #recognizer
        id_, conf = recognizer.predict(roi_gray)
        if conf<=10 or conf>=90: # tolerance?
            print(labels[id_], round(conf, 1))
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (0,0,255)
            stroke = 2
            cv2.putText(frame, name, (x_start,y_start-8), font, 0.8, color, stroke, cv2.LINE_AA)
            

        color = (255,0,0)
        stroke = 2
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, stroke)


    cv2.imshow("frame", frame)
    if cv2.waitKey(20) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
