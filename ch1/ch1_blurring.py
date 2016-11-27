from PIL import Image
from pylab import *
from scipy.ndimage import filters

im = array(Image.open('./images/empire.jpg').convert('L'))
im2 = filters.gaussian_filter(im, 5)

figure()
gray()
imshow(im2)
show()