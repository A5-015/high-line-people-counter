from __future__ import print_function
import cv2 as cv
import argparse

## [create]
#create Background Subtractor objects
#backSub = cv.createBackgroundSubtractorMOG2()

# 	cv.createBackgroundSubtractorKNN(	[, history[, dist2Threshold[, detectShadows]]]	)
backSub = cv.createBackgroundSubtractorKNN([, 50[, 400.0[, True]]]


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)


while True:

	capture = cv.VideoCapture("http://192.168.1.2/cam-hi.jpg")

	ret, frame = capture.read()

	#rescale frame
	frame = rescale_frame(frame, percent=50)

	#update the background model
	fgMask = backSub.apply(frame)

	#get the frame number and write it on the current frame
	cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
	cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
	           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

	#show the current frame and the fg masks
	cv.imshow('Frame', frame)
	cv.imshow('FG Mask', fgMask)

	keyboard = cv.waitKey(500)
