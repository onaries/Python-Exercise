from PIL import Image
import numpy
import glob

from PCV.geometry import camera

# load some images
im1 = numpy.array(Image.open('data/images/001.jpg'))
im2 = numpy.array(Image.open('data/images/002.jpg'))

# load 2D points for each view to a list
points2D = [numpy.loadtxt(f).T for f in glob.glob('data/2D/*.corners')]

# loat 3D points
points3D = numpy.loadtxt('data/3D/p3d').T

# load correspondences
corr = numpy.genfromtxt('data/2D/nview-corners', dtype='int', missing_values='*')

# load cameras to a list of Camera objects
P = [camera.Camera(numpy.loadtxt(f)) for f in glob.glob('data/2D/*.P')]
