# Criteria, defining object points
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# IMPORTANT: Enter chess board dimensions
chw = 9
chh = 6


# Defining draw functions for lines
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


# Load previously saved data
with np.load('B.npz') as X:
    mtx, dist, _, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs', 'imgpts')]

# Criteria, defining object points
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chh * chw, 3), np.float32)
objp[:, :2] = np.mgrid[0:chw, 0:chh].T.reshape(-1, 2)

# Setting axis
axis = np.float32([[9, 0, 0], [0, 6, 0], [0, 0, -10]]).reshape(-1, 3)

cap = cv2.VideoCapture('Calibration\\video_chess2.MP4')
count = 0
fcount = 0
while (cap.isOpened()):
    ret1, img = cap.read()
    if ret1 == False or count == lim:
        print('Video analysis complete.')
        break
    if count > 0:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # Undistorting
        img2 = cv2.undistort(img, mtx, dist, None, newcameramtx)

        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret2, corners = cv2.findChessboardCorners(gray, (chw, chh), None)
        if ret2 == True:
            fcount = fcount + 1
            # Refining corners
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors
            rvecs_new, tvecs_new, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

            # Project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs_new, tvecs_new, mtx, dist)
            draw(img2, corners, imgpts)
            cv2.imshow('img', img2)
            cv2.waitKey(1)

            # Converting rotation vector to rotation matrix
            np_rodrigues = np.asarray(rvecs_new[:, :], np.float64)
            rmatrix = cv2.Rodrigues(np_rodrigues)[0]

            # Pose (According to http://stackoverflow.com/questions/16265714/camera-pose-estimation-opencv-pnp)
            cam_pos = -np.matrix(rmatrix).T * np.matrix(tvecs_new)

            camx.append(cam_pos.item(0))
            camy.append(cam_pos.item(1))
            camz.append(cam_pos.item(2))

        else:
            print 'Board not found'

    count += 1
    print count
cv2.destroyAllWindows()
plt.plot(camx, camy)
plt.show()