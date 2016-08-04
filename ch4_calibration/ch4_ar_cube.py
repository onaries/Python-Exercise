from pylab import *
from numpy import *
from PIL import Image

# If you have PCV installed, these imports should work
from PCV.geometry import homography, camera
from PCV.localdescriptors import sift

"""
This is the augmented reality and pose estimation cube example from Section 4.3.
"""

def cube_points(c,wid):
    """ Creates a list of points for plotting
        a cube with plot. (the first 5 points are
        the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]-wid,c[2]-wid]) #same as first to close plot
    
    # top
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]-wid,c[2]+wid]) #same as first to close plot
    
    # vertical sides
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    
    return array(p).T


def my_calibration(sz):
    """
    Calibration function for the camera (iPhone4) used in this example.
    """
    # row,col = sz
    row, col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    # K[0, 2] = 0.5 * row
    # K[1, 2] = 0.5 * col
    return K

def my_calibration2(sz):
    """
    Calibration function for the camera (iPhone4) used in this example.
    """
    # row,col = sz
    row, col = sz
    fx = 3538*col/4032
    fy = 3605*row/2268
    K = diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    # K[0, 2] = 0.5 * row
    # K[1, 2] = 0.5 * col
    return K



# compute features
# sift.process_image('./data/book_frontal.JPG','./data/im0.sift')
l0,d0 = sift.read_features_from_file('./data/im0.sift')

# sift.process_image('./data/book_perspective.JPG','./data/im1.sift')
l1,d1 = sift.read_features_from_file('./data/im1.sift')

# sift.process_image('./data/image1.JPG', './data/im2.sift')
# l0,d0 = sift.read_features_from_file('./data/im2.sift')
#
# sift.process_image('./data/image2.JPG', './data/im3.sift')
# l1,d1 = sift.read_features_from_file('./data/im3.sift')

# sift.process_image('./data/s20160720_113416.JPG', './data/im2.sift')
# l0,d0 = sift.read_features_from_file('./data/im2.sift')
#
# sift.process_image('./data/s20160720_113436.JPG', './data/im3.sift')
# l1,d1 = sift.read_features_from_file('./data/im3.sift')

# match features and estimate homography
matches = sift.match_twosided(d0,d1)

# im0 = array(Image.open('./data/book_frontal.JPG'))
# im1 = array(Image.open('./data/book_perspective.JPG'))
# im0 = array(Image.open('./data/s20160720_113416.JPG'))
# im1 = array(Image.open('./data/s20160720_113436.JPG'))
# figure()
# sift.plot_matches(im0,im1,l0,l1, matches, show_below=True)
# show()

ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx,:2].T)

ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2,:2].T)

# x axis change y axis
fp = array([fp[1,:],fp[0,:],fp[2,:]])
tp = array([tp[1,:],tp[0,:],tp[2,:]])

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp,tp,model)

# camera calibration
K = my_calibration((747,1000))

print K

# 3D points at plane z=0 with sides of length 0.2
# box = cube_points([0.15,-0.15,0.1],0.1)
box = cube_points([0,0,0.1],0.09)

# project bottom square in first image
cam1 = camera.Camera( hstack((K,dot(K,array([[0],[0],[-1]])) )) )
# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:,:5]))


# use H to transfer points to the second image
box_trans = homography.normalize(dot(H,box_cam1))

# compute second camera matrix from cam1 and H
cam2 = camera.Camera(dot(H,cam1.P))
A = dot(linalg.inv(K),cam2.P[:,:3])
A = array([A[:,0],A[:,1],cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = dot(K,A)

# print cam2.P
Rt = dot(linalg.inv(K),cam2.P)
print Rt

# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))

point = array([1,1,0,1]).T
print homography.normalize(dot(dot(H,cam1.P), point))
print cam2.project(point)

# plotting
im0 = array(Image.open('./data/book_frontal.JPG'))
im1 = array(Image.open('./data/book_perspective.JPG'))
# im0 = array(Image.open('./data/s20160720_113416.JPG'))
# im1 = array(Image.open('./data/s20160720_113436.JPG'))

# figure()
# imshow(im0)
# plot(box_cam1[1,:],box_cam1[0,:],linewidth=3)
# plot(fp[1,:],fp[0,:],'k.')
# title('2D projection of bottom square')
# axis('off')
#
# figure()
# imshow(im1)
# plot(box_trans[1,:],box_trans[0,:],linewidth=3)
# title('2D projection transfered with H')
# axis('off')
#
# figure()
# imshow(im1)
# plot(box_cam2[1,:],box_cam2[0,:],linewidth=3)
# title('3D points projected in second image')
# axis('off')

figure()
imshow(im0)
plot(box_cam1[0,:],box_cam1[1,:],linewidth=3)
title('2D projection of bottom square')
axis('off')

figure()
imshow(im1)
plot(box_trans[0,:],box_trans[1,:],linewidth=3)
title('2D projection transfered with H')
axis('off')

figure()
imshow(im1)
plot(box_cam2[0,:],box_cam2[1,:],linewidth=3)
title('3D points projected in second image')
axis('off')

import pickle
with open('ar_camera.pkl', 'w') as f:
    pickle.dump(K,f)
    pickle.dump(dot(linalg.inv(K),cam2.P),f)
    # pickle.dump(array([[-0.29453529, -0.99008224, -0.0355475,  -0.17885403],[ 0.26348232, -0.26594492,  0.28379332,  0.18004378],[-0.25131098,  0.11874566, 0.33919933, -0.91711127]]),f)

show()