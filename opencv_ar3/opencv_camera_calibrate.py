import cv2
import pylab
import numpy as np
import glob

def my_calibration2(sz):
    """
    Calibration function for the camera (iPhone4) used in this example.
    """
    # row,col = sz
    row, col = sz
    fx = 3538*col/4032
    fy = 3605*row/2268
    K = pylab.diag([fx,fy,1])
    K[0,2] = 0.5*col
    K[1,2] = 0.5*row
    # K[0, 2] = 0.5 * row
    # K[1, 2] = 0.5 * col
    return K

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret = False
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6))
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print ret, mtx, dist, rvecs, tvecs
cv2.destroyAllWindows()