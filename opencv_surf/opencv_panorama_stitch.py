# import the necessary packages
from opencv_panorama import Stitcher
import argparse
import imutils
import cv2
import glob
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True,
#                 help="path to the first image")
# ap.add_argument("-s", "--second", required=True,
#                 help="path to the second image")
# args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)

# im1 = './data/images/bryce_left_02.png'
# im2 = './data/images/bryce_right_02.png'

im1 = './data/images/grand_canyon_left_02.png'
im2 = './data/images/grand_canyon_right_02.png'
# imlists = glob.glob('./data/images/sift*')
# print imlists

# imageA = cv2.imread(imlists[0])
# imageB = cv2.imread(imlists[1])
# imageC = cv2.imread(imlists[2])
# imageD = cv2.imread(imlists[3])
# imageE = cv2.imread(imlists[4])
# imageA = imutils.resize(imageA, width=400)
# imageB = imutils.resize(imageB, width=400)
# imageC = imutils.resize(imageC, width=400)
# imageD = imutils.resize(imageD, width=400)
# imageE = imutils.resize(imageE, width=400)

imageA = cv2.imread(im1)
imageB = cv2.imread(im2)
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
# (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
# (result, vis) = stitcher.stitch([result, imageC], showMatches=True)
# (result, vis) = stitcher.stitch([result, imageD], showMatches=True)
# (result, vis) = stitcher.stitch([result, imageE], showMatches=True)

# show the images
# cv2.imshow("Image A", imageA)
# cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
