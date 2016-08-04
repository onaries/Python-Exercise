from pylab import *
from numpy import *
from PIL import Image
import cv2

# If you have PCV installed, these imports should work
from PCV.geometry import homography, camera
from PCV.localdescriptors import sift

"""
This is the augmented reality and pose estimation cube example from Section 4.3.
"""


def cube_points(c, wid):
    """ Creates a list of points for plotting
        a cube with plot. (the first 5 points are
        the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])  # same as first to close plot

    # top
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])  # same as first to close plot

    # vertical sides
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])

    return array(p).T


def my_calibration(sz):
    """
    Calibration function for the camera (iPhone4) used in this example.
    """
    # row,col = sz
    row, col = sz
    fx = 2555 * col / 2592
    fy = 2586 * row / 1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5 * col
    K[1, 2] = 0.5 * row
    # K[0, 2] = 0.5 * row
    # K[1, 2] = 0.5 * col
    return K


# compute features
# sift.process_image('./data/book_frontal.JPG','./data/im0.sift')
# l0, d0 = sift.read_features_from_file('./data/im0.sift')

# sift.process_image('./data/book_perspective.JPG','./data/im1.sift')
# l1, d1 = sift.read_features_from_file('./data/im1.sift')

# sift.process_image('./data/image1.JPG', './data/im2.sift')
# l0,d0 = sift.read_features_from_file('./data/im2.sift')
#
# sift.process_image('./data/image2.JPG', './data/im3.sift')
# l1,d1 = sift.read_features_from_file('./data/im3.sift')

def match_images(img1, img2):

    detector = cv2.SURF(400, 5, 5)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)

    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return kp1, kp2, desc1, desc2, kp_pairs

def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs

def explore_match(win, img1, img2, kp_pairs, status=None, H=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = zeros((max(h1, h2), w1 + w2), uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = ones(len(kp_pairs), bool_)
    p1 = int32([kpp[0].pt for kpp in kp_pairs])
    p2 = int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)


def draw_matches(window_name, kp_pairs, img1, img2):
    """Draws the matches for """
    mkp1, mkp2 = zip(*kp_pairs)

    p1 = float32([kp.pt for kp in mkp1])
    p2 = float32([kp.pt for kp in mkp2])

    if len(kp_pairs) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        # print '%d / %d  inliers/matched' % (numpy.sum(status), len(status))
    else:
        H, status = None, None
        # print '%d matches found, not enough for homography estimation' % len(p1)

    if len(p1):
        explore_match(window_name, img1, img2, kp_pairs, status, H)

# match features and estimate homography

fn1 = './data/s20160720_113416.jpg'
fn2 = './data/s20160720_113436.jpg'

img1 = cv2.imread(fn1, 0)
img2 = cv2.imread(fn2, 0)

l0, l1, d0, d1, kp_pairs = match_images(img1, img2)

l0 = array(l0)
l1 = array(l1)
print l0
print d0

matches = sift.match_twosided(d0, d1)

# im0 = array(Image.open('./data/book_frontal.JPG'))
# im1 = array(Image.open('./data/book_perspective.JPG'))
# im0 = array(Image.open('./data/s20160720_113416.JPG'))
# im1 = array(Image.open('./data/s20160720_113436.JPG'))
# figure()
# sift.plot_matches(im0,im1,l0,l1, matches, show_below=True)
# show()

ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)

ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)

# x axis change y axis
fp_tmp = array([fp[1, :], fp[0, :], fp[2, :]]);
fp = fp_tmp
tp_tmp = array([tp[1, :], tp[0, :], tp[2, :]]);
tp = tp_tmp

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp, tp, model)

# camera calibration
K = my_calibration((747, 1000))

print K

# 3D points at plane z=0 with sides of length 0.2
# box = cube_points([0.15,-0.15,0.1],0.1)
box = cube_points([0, 0, 0.1], 0.09)

# project bottom square in first image
cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

# use H to transfer points to the second image
box_trans = homography.normalize(dot(H, box_cam1))

# compute second camera matrix from cam1 and H
cam2 = camera.Camera(dot(H, cam1.P))
A = dot(linalg.inv(K), cam2.P[:, :3])
A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)

print cam2.P
Rt = dot(linalg.inv(K), cam2.P)
print Rt

# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))

# plotting
im0 = array(Image.open('./data/book_frontal.JPG'))
im1 = array(Image.open('./data/book_perspective.JPG'))
# im0 = array(Image.open('./data/image1.JPG'))
# im1 = array(Image.open('./data/image2.JPG'))

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
plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)
title('2D projection of bottom square')
axis('off')

figure()
imshow(im1)
plot(box_trans[0, :], box_trans[1, :], linewidth=3)
title('2D projection transfered with H')
axis('off')

figure()
imshow(im1)
plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)
title('3D points projected in second image')
axis('off')

import pickle

with open('ar_camera.pkl', 'w') as f:
    pickle.dump(K, f)
    pickle.dump(dot(linalg.inv(K), cam2.P), f)
    # pickle.dump(array([[-0.29453529, -0.99008224, -0.0355475,  -0.17885403],[ 0.26348232, -0.26594492,  0.28379332,  0.18004378],[-0.25131098,  0.11874566, 0.33919933, -0.91711127]]),f)

show()