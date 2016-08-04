from numpy import *
from pylab import *
import PIL

from PCV.localdescriptors import sift

imname = './data/empire.jpg'
im1 = array(Image.open(imname).convert('L'))