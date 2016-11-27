from PIL import Image
from pylab import *
from PCV.tools import imtools

im = array(Image.open('./images/empire.jpg').convert('L'))
im2, cdf = imtools.histeq(im)
figure()
hist(im.flatten(),128)
title('Before the Histogram Equalization')
figure()
title('After the Histogram Equalization')
hist(im2.flatten(),128)
show()