import numpy as np
import cv2
import os # check file, dir
import shutil # delete a directory

face_cascade = cv2.CascadeClassifier('./data/cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./data/cascade/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('./data/cascade/haarcascade_mouth.xml')
nose_cascade = cv2.CascadeClassifier('./data/cascade/haarcascade_mcs_nose.xml')
lefteye_cascade = cv2.CascadeClassifier('./data/cascade/haarcascade_lefteye_2splits.xml')
righteye_cascade = cv2.CascadeClassifier('./data/cascade/haarcascade_righteye_2splits.xml')

img = cv2.imread('./data/face/SeonwooKim.jpg')
img2 = img.copy() # image copy
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

i = 0
if os.path.isdir('./data/feature'):
    shutil.rmtree('./data/feature')
os.mkdir('./data/feature')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces: # w = width, h = height
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(192,0,0),2)
    # save the face
    crop_face = img2[y:y+h, x:x+w]
    cv2.imwrite('./data/feature/face' + str(i + 1) + '.jpg',crop_face)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # left eye detection
    lefteye = lefteye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in lefteye:
        if ey < (y / 2):
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # crop the left eye and save the image
            crop_lefteye = img2[y+ey:y+ey+eh, x+ex:x+ex+ew]
            cv2.imwrite('./data/feature/lefteye' + str(i + 1) + '.jpg',crop_lefteye)

    # right eye detection
    righteye = righteye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in righteye:
        if ey < (y / 2):
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # crop the right eye and save the image
            crop_righteye = img2[y+ey:y+ey+eh, x+ex:x+ex+ew]
            cv2.imwrite('./data/feature/righteye' + str(i + 1) + '.jpg',crop_righteye)

    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # nose = nose_cascade.detectMultiScale(roi_gray)
    # for (nx, ny, nw, nh) in nose:
    #     break
        #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

    # mouth detection
    mouth = mouth_cascade.detectMultiScale(roi_gray)
    j = 0
    for (mx, my, mw, mh) in mouth:
        #if (my > ny + (nh / 2)):
        if (my > (w / 2)):  # w = face width
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
            # crop the mouth and save the image
            crop_mouth = img2[y+my:y+my+mh, x+mx:x+mx+mw]
            if os.path.isfile('./data/feature/mouth' + str(j + 1) + '.jpg'):
                j += 1
            cv2.imwrite('./data/feature/mouth' + str(j + 1) + '.jpg',crop_mouth)

    i += 1





cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

