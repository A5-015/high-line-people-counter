from __future__ import print_function
import cv2
from scipy.ndimage.filters import gaussian_filter

## [create]
#create Background Subtractor objects
#backSub = cv2.createBackgroundSubtractorMOG2()

# cv2.createBackgroundSubtractorKNN(	[, history[, dist2Threshold[, detectShadows]]]	)
# cv2.createBackgroundSubtractorKNN(50, 400.0, True)
backSub = cv2.createBackgroundSubtractorKNN(50, 400.0, True)


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


while True:

    capture = cv2.VideoCapture("http://192.168.1.2/cam-hi.jpg")

    ret, frame = capture.read()

    #rescale frame
    frame = rescale_frame(frame, percent=50)
    frameBlurred = gaussian_filter(frame, sigma=2)

    #update the background model
    fgMask = backSub.apply(frameBlurred)
    fgMask = gaussian_filter(fgMask, sigma=2)

    #get the frame number and write it on the current frame
    #cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
    #cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    x, y, w, h = cv2.boundingRect(fgMask)
    rect1 = cv2.rectangle(frame.copy(),(x,y),(x+w,y+h),(0,255,0),3) # not copying here will throw an e

    #show the current frame and the fg masks
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    cv2.imshow('rect', rect1)

    keyboard = cv2.waitKey(500)
