from PIL import Image
from numpy import *
from pylab import *
from PCV.tools import pca
import glob

file_list = glob.glob('./data/face/2008*.jpg')

im = array(Image.open(file[0])) # open one image to get size
m,n = im.shape[0:2] # get the size of the images
imnbr = len(file_list) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(file)).flatten() for file in file_list], 'f')

# perform PCA
V,S,immean = pca.pca(immatrix)



imsave('./data/feature')
