import numpy as np
import cv2
import sys

def match_images(img1, img2):
    detector = cv2.xfeatures2d.SURF_create(400, 5, 5)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)

    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return kp_pairs

def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs

