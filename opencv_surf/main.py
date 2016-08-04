import cv2
import numpy as np

img = cv2.imread('./data/s20160720_113416.jpg',0)
imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

surf = cv2.SURF()

kp,des = surf.detect(imgg,None,useProvidedKeyPoints = False)

samples = np.array(des)
responses = np.arange(len(kp),dtype=np.float32)

knn = cv2.KNearest()
knn.train(samples, responses)



len(kp)