from PIL import Image
import numpy
import glob

from PCV.geometry import camera

im1 = numpy.array(Image.open('data/images/001.jpg'))
im2 = numpy.array(Image.open('data/images/002.jpg'))

points2D = [numpy.loadtxt(f).T for f in glob.glob('data/2D/*.corners')]
points3D = numpy.loadtxt('data/3D/p3d').T

corr = numpy.genfromtxt('data/2D/nview-corners', dtype='int', missing_values='*')

P = [camera.Camera(numpy.loadtxt(f)) for f in glob.glob('data/2D/*.P')]
