import cv2
import numpy as np 
import sqlite3
import os

conn = sqlite3.connect('database.db')
c = conn.cursor()

fname = "recognizer/trainingData.yml"
if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)

face_cascade = cv2.CascadeClassifier('cascadeClassifiers/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)
authentication_score = 0

while True:
  ret, img = cap.read()
  if authentication_score < 50:
    color = (0,0,255)
  else:
    color = (150,255,0)
  cv2.putText(img, "Authentication score: %s" %authentication_score, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  if len(faces) < 1:
    authentication_score -= 1
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
    c.execute("select name from users where id = (?);", (ids,))
    result = c.fetchall()
    name = result[0][0]
    if conf < 50:
      cv2.putText(img, "%s : Confidence %s" %(name, round(conf, 2)), (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)
      authentication_score = 100
    else:
      cv2.putText(img, 'No Match', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)
      authentication_score = authentication_score - 10
  cv2.imshow('Face Recognizer',img)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break

cap.release()
cv2.destroyAllWindows()