from pylab import *
from numpy import *
from PIL import Image
from PCV.geometry import camera, homography
from PCV.localdescriptors import sift

# compute features
sift.process_image('./data/book_frontal.JPG', './data/im0.sift')
l0,d0 = sift.read_features_from_file('./data/im0.sift')

sift.process_image('./data/book_perspective.JPG', './data/im1.sift')
l1,d1 = sift.read_features_from_file('./data/im1.sift')

# sift.process_image('./data/s20160720_113416.JPG', './data/im2.sift')
# l0,d0 = sift.read_features_from_file('./data/im2.sift')
#
# sift.process_image('./data/s20160720_113436.JPG', './data/im3.sift')
# l1,d1 = sift.read_features_from_file('./data/im3.sift')

# match features and estimate homography
matches = sift.match_twosided(d0,d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx,:2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2,:2].T)

model = homography.RansacModel()
H = homography.H_from_ransac(fp,tp,model)

