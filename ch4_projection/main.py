from pylab import *
from numpy import *
from PIL import Image
from PCV.geometry import camera

# load points
points = loadtxt('./data/house.p3d').T
points = vstack((points, ones(points.shape[1])))

# setup camera
P = hstack((eye(3), array([[0], [0], [-10]])))
cam = camera.Camera(P)
x = cam.project(points)

# plot projection
figure()
plot(x[0], x[1], 'k.')
show()
