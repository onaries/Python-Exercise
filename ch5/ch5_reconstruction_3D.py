'''
p146 3D reconstruction example
'''

from PIL import Image
from pylab import *
import sfm
from PCV.geometry import camera, homography
from PCV.localdescriptors import sift
import tic

# calibration
K = array([[2394,0,932],[0,2398,628],[0,0,1]])

tic.k('start')
# load images and compute featuers
im1 = array(Image.open('./images/salcatraz1.jpg'))
sift.process_image('./images/salcatraz1.jpg','./images/salcatraz1.sift')
l1,d1 = sift.read_features_from_file('./images/salcatraz1.sift')

im2 = array(Image.open('./images/salcatraz2.jpg'))
sift.process_image('./images/salcatraz2.jpg','./images/salcatraz2.sift')
l2,d2 = sift.read_features_from_file('./images/salcatraz2.sift')

tic.k('loadd sifts')
print '{} / {} features'.format(len(d1), len(d2))

# match features
matches = sift.match_twosided(d1,d2)
ndx = matches.nonzero()[0]

tic.k('matched')

# make homogeneous and normalize with inv(K)
x1 = homography.make_homog(l1[ndx,:2].T)
x1 = array([x1[1,:],x1[0,:],x1[2,:]])
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2,:2].T)
x2 = array([x2[1,:],x2[0,:],x2[2,:]])

x1n = dot(inv(K),x1)
x2n = dot(inv(K),x2)

tic.k('normalized')

# estimate E with RANSAC
model = sfm.RansacModel()
E,inliers = sfm.F_from_ransac(x1n,x2n,model)

tic.k('ransacd, %d inliers' % len(inliers))

# compute camera matrices (P2 will be list of four solutions)
P1 = array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P2 = sfm.compute_P_from_essential(E)

tic.k('computed possible camera matrices')

# pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
    # triangulation inliers and compute depth for each camera
    X = sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[i])
    d1 = dot(P1,X)[2]
    d2 = dot(P2[i],X)[2]

    if sum(d1>0)+sum(d2>0) > maxres:
        maxres = sum(d1>0)+sum(d2>0)
        ind = i
        infront = (d1>0) & (d2>0)


tic.k('picked one')

# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:,inliers],x2n[:,inliers],P1,P2[ind])
X = X[:,infront]

tic.k('triangulated')

# 3D plot
from mpl_toolkits.mplot3d import axes3d

fig = figure()
ax = fig.gca(projection='3d')
ax.plot(-X[0],X[1],X[2],'k.')
axis('off')

# project 3D points
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[ind])
x1p = cam1.project(X)
x2p = cam2.project(X)

# reverse K normalization
x1p = dot(K,x1p)
x2p = dot(K,x2p)

figure()
imshow(im1)
gray()
plot(x1p[0],x1p[1],'o')
plot(x1[0],x1[1],'r.')
axis('off')

figure()
imshow(im2)
gray()
plot(x2p[0],x2p[1],'o')
plot(x2[0],x2[1],'r.')
axis('off')
show()