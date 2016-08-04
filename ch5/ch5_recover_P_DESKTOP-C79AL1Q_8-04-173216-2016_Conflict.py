'''
p139~141 Computing the camera matrix from 3D points
'''

from pylab import *
import sfm
from PCV.geometry import camera

execfile('load_vggdata.py')

corr = corr[:, 0]   # view 1
ndx3D = numpy.where(corr >= 0)[0]   # missing values are -1
ndx2D = corr[ndx3D]

# select visible points and makehomogeneous
x = points2D[0][:, ndx2D]
x = numpy.vstack( (x, numpy.ones(x.shape[1])) )
X = points3D[:, ndx3D]
X = numpy.vstack( (X, numpy.ones(X.shape[1])) )

# estimate P
Pest = camera.Camera(sfm.compute_P(x,X))

# Check:
print Pest.P / Pest.P[2, 3]
print P[0].P / P[0].P[2, 3]

xest = Pest.project(X)

# plotting
figure()
imshow(im1)
plot(x[0], x[1], 'bo')
plot(xest[0], xest[1], 'r.')
axis('off')

show()

