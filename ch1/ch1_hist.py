from PIL import Image
from pylab import *

im = array(Image.open('./images/empire.jpg').convert('L'))

# create a new figure
figure()

# don't use colors
gray()

# show contours with origin upper left corner
contour(im, origin='image')
axis('equal')
axis('off')

figure()
hist(im.flatten(),128)
show()