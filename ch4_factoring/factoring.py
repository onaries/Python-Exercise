from pylab import *
from numpy import *
from PIL import Image
from PCV.geometry import camera

K = array([[1000,0,500],[0,1000,300],[0,0,1]])
tmp = camera.rotation_matrix([0,0,1])[:3,:3]
Rt = hstack((tmp, array([[50], [40], [30]])))
cam = camera.Camera(dot(K,Rt))

print K
print Rt
print cam.factor()