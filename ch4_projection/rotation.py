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

# create transformation
r = 0.05*random.rand(3)
rot = camera.rotation_matrix(r)

# rotate camera and project
figure()
for t in range(20):
    cam.P = dot(cam.P, rot)
    x = cam.project(points)
    plot(x[0], x[1], 'k.')
show()