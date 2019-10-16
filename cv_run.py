import numpy as np
import cv2
import os

import time
timestr = time.strftime("%Y%m%d__%H%M%S")
path = "image/__"+timestr+"/"

if not os.path.exists(path):
    os.makedirs(path)

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# HACKTOBERFEST ? PULL REQ1 :) vasi why not remove all commented code?
# while 1:
    # ret, img = cap.read()
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
k = cv2.waitKey(0)
# if k == 27:
#     break

i=0

print("Detected faces:",len(faces))
for (x, y, w, h) in faces:
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)

    faceimg = img[ny:ny+nr, nx:nx+nr]
    lastimg = cv2.resize(faceimg, (32, 32))
    i += 1
    cv2.imwrite("path/image%d.jpg" % i, lastimg)

cap.release()
cv2.destroyAllWindows()
