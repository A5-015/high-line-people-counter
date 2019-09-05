from __future__ import print_function
import cv2
from scipy.ndimage.filters import gaussian_filter
import numpy as np

## [create]
#create Background Subtractor objects
#backSub = cv2.createBackgroundSubtractorMOG2()

# cv2.createBackgroundSubtractorKNN(	[, history[, dist2Threshold[, detectShadows]]]	)
# cv2.createBackgroundSubtractorKNN(50, 400.0, True)
#backSub = cv2.createBackgroundSubtractorKNN(50, 300.0, False)
backSub = cv2.createBackgroundSubtractorKNN()

peopleCount = 0

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


while True:
    capture = cv2.VideoCapture("http://192.168.1.2/cam-hi.jpg")

    ret, frame = capture.read()

    # Rescale frame
    frame = rescale_frame(frame, percent=50)

    # Add blur
    frameBlurred = gaussian_filter(frame, sigma=1)

    #update the background model and add blur
    foregroundMask = backSub.apply(frameBlurred)
    foregroundMask = gaussian_filter(foregroundMask, sigma=2)

    # Find individual contours
    contours, hier = cv2.findContours(foregroundMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        # skip the rectangle if falls into certain coordinates

        if not (((x > 170) and (y < 260)) or (x > 150) and (x < 620) and (y < 430)):
            cv2.rectangle(frame, (x-5, y-5), (x+w+10, y+h+10), (0, 255, 0), 1)

            '''
            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            # draw a red 'nghien' rectangle
            cv2.drawContours(frame, [box], 0, (0, 0, 255))

            # finally, get the min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            # convert all values to int
            center = (int(x), int(y))
            radius = int(radius)
            # and draw the circle in blue
            frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)
            '''
            peopleCount = peopleCount + 1

    # Draw the people counter
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(peopleCount), (15, 15),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    # Show images
    cv2.imshow('FG Mask', foregroundMask)
    cv2.imshow('Frame', frame)

    keyboard = cv2.waitKey(100)
