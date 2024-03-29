# import the necessary packages
from __future__ import print_function
import numpy as np
import urllib.request as urllib
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import time

while True:

    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen("http://192.168.1.2/cam-hi.jpg")
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    #image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.imdecode(image, -1)


    #image = cv2.imread("/Users/bsimsek/Desktop/cam.png")

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = imutils.resize(image, width=min(800, image.shape[1]))

    # crop the image
    image = image[170:800, 0:140]

    orig = image.copy()

    # detect people in the image
    #(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), scale=1.03)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
    	cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
    	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)


    # show some information on the number of bounding boxes
    #filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: original boxes, {} after suppression".format(len(rects), len(pick)))

    # show the output images
    cv2.imshow('Highline', image)
    cv2.waitKey(200)
    #cv2.destroyAllWindows()
