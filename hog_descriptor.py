# performs pedestrian detection on still images based on opencv built-in HOG + SVM method

# the picture directory
pic_dir = "C:\\Users\\User\\Pictures\\pedestrian.jpg"

# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# load the image
img = cv2.imread(pic_dir)
# reduce image size to reduce detection time and improve detection accuracy
#img = imutils.resize(img, width=min(400, img.shape[1]))
orig = img.copy()

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

import time
tic = time.time()
# detect people in the image, weights: confidence value returned by svm for each detection
(rects, weights) = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8),\
                                        scale = 1.02)
#print('detecting confidence:{}'.format(weights))

print("time elapsed for hog: {}".format(time.time()-tic))


# draw the original bounding boxes
for (x, y, w, h) in rects:
    cv2.rectangle(orig, (x,y), (x+w, y+h), (0, 0, 255), 2)


# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)


# show some information on the number of bounding boxes
filename = pic_dir[pic_dir.rfind("/") + 1:]
print("[INFO] {}: {} original boxes, {} after suppression".format(\
		filename, len(rects), len(pick)))


# show the output images
cv2.imshow("Before NMS", orig)
cv2.imshow("After NMS", img)
cv2.waitKey(0)
