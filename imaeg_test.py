# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:45:54 2020

@author: sayya
"""

import numpy as np
import cv2
import tensorflow as tf

face_detection = cv2.CascadeClassifier(r'C:\Users\sayya\OneDrive\Desktop\facial-expression-recognition-master\haar_cascade_face_detection.xml')

#camera = cv2.VideoCapture(0)
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
settings = {
    'scaleFactor': 1.3, 
    'minNeighbors': 5, 
    'minSize': (50, 50)
}

labels = ["Neutral","Happy","Sad","Surprise","Angry"]

model = tf.keras.models.load_model(r'C:\Users\sayya\OneDrive\Desktop\facial-expression-recognition-master\expression.model')


	#ret, img = camera.read()
img=cv2.imread(r"C:\Users\sayya\OneDrive\Pictures\Camera Roll\faizan.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detected = face_detection.detectMultiScale(gray, **settings)
    
for x, y, w, h in detected:
	cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
	cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
	face = gray[y+5:y+h-5, x+20:x+w-20]
	face = cv2.resize(face, (48,48)) 
	face = face/255.0
		
	predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
	state = labels[predictions]
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img,state,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
		
cv2.imshow('Facial Expression', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#writer = cv2.VideoWriter()
#writer.write(img)
#if cv2.waitKey(5) != -1:
#	break

cv2.waitKey(0)

#camera.release()
cv2.destroyAllWindows()
