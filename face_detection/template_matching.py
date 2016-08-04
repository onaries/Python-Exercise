import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./data/face/m01-32_gr.jpg',0)
img2 = img.copy()
template = cv2.imread('./data/feature/face.jpg',0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

match_list1 = []
match_list2 = []
match_num_1 = 0
match_num_2 = 0

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    match_list1.append(top_left)
    match_list2.append(bottom_right)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    # plt.show()

for i in range(match_list1.__len__() - 1):
    if match_list1[i] == match_list1[i+1]:
        match_num_1 += 1
    if match_list2[i] == match_list2[i+1]:
        match_num_2 += 1

if match_num_1 > 2 and match_num_2 > 2:
    print 'face detect'
else:
    print 'face not detect'

