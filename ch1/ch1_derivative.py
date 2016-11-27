from PIL import Image
from pylab import *
from scipy.ndimage import filters

im = array(Image.open('./images/empire.jpg').convert('L'))

# Sobel derivative filters
imx = zeros(im.shape)
filters.sobel(im,1,imx)

imy = zeros(im.shape)
filters.sobel(im,0,imy)

magnitude = sqrt(imx**2+imy**2)

figure()
title('x axis derivation')
gray()
imshow(imx)
figure()
title('y axis derivation')
gray()
imshow(imy)
show()