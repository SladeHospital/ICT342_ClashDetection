import cv2
import numpy as np
import math


def trackbar_onChange(a):
    print(a)
    pass


def prepare_frames():
    global frameIndex
    global videoIndex
    frameIndex = 0
    framesArray.clear()
    cap = cv2.VideoCapture(footageDir + "dashcam_footage_%d.avi" % videoIndex)
    extracting = True
    while extracting:
        extracting, currentFrame = cap.read()
        if extracting:
            framesArray.append(currentFrame)

    print(len(framesArray))

    videoIndex += 1


def stack_images(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


footageDir = "ClashDetectionMaterial/"

rightArrowCode = 2555904
leftArrowCode = 2424832
videoIndex = 1

kernel = np.ones((5, 5), np.uint8)
mouseX, mouseY = 0, 0
framesArray = []
frameIndex = 0
success = True


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Gauss Blur", "TrackBars", 1, 21, trackbar_onChange)
cv2.createTrackbar("Canny Min", "TrackBars", 100, 500, trackbar_onChange)
cv2.createTrackbar("Canny Max", "TrackBars", 200, 500, trackbar_onChange)
prepare_frames()


while True:
    currentFrame = framesArray[0]
    while frameIndex <= len(framesArray):
        currentFrame = framesArray[frameIndex]

        grayImg = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
        gaussDim = cv2.getTrackbarPos('Gauss Blur', 'TrackBars')

        # Checks to see if the value of the trackbar is even as the gaussian function doesn't work with even dimension
        # Skips even numbers by setting the position of the trackbar to the next odd number
        # This is fucked but seems like the only solution other than not using trackbars
        count = gaussDim % 2
        if count == 0:
            gaussDim += 1
            cv2.setTrackbarPos('Gauss Blur', 'TrackBars', gaussDim)
            gaussImg = cv2.GaussianBlur(grayImg, (gaussDim, gaussDim), 0)
            print(gaussDim)
        else:
            gaussImg = cv2.GaussianBlur(grayImg, (gaussDim, gaussDim), 0)
            print(gaussDim)

        cannyMin = cv2.getTrackbarPos('Canny Min', 'TrackBars')
        cannyMax = cv2.getTrackbarPos('Canny Max', 'TrackBars')
        edgesImg = cv2.Canny(gaussImg, cannyMin, cannyMax)
        imgStack = stack_images(0.6, ([currentFrame, gaussImg], [edgesImg, currentFrame]))
        cv2.imshow("dashcam_footage", imgStack)

        # cv2.waitKey waits for a keypress or its parameter in ms, whichever comes first
        # unless the parameter is 0, then it will wait for the keypress forever
        key = cv2.waitKeyEx(1)
        if key == rightArrowCode or key == ord('d'):
            if frameIndex >= len(framesArray)-1:
                prepare_frames()
            else:
                frameIndex += 1
                print(frameIndex)

        if (key == leftArrowCode or key == ord('a')) and frameIndex > 0:
            frameIndex -= 1

        # Quit when 'q' is pressed
        if key == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()