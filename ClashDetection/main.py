import cv2
import numpy as np
import math


def trackbar_onChange(a):
    pass


def prepare_frames():
    global frameIndex
    global videoIndex
    global footageIndex
    frameIndex = 0
    framesArray.clear()
    cap = cv2.VideoCapture(footageDir + "dashcam_footage_%d.avi" % footageIndex)
    extracting = True
    while extracting:
        extracting, currentFrame = cap.read()
        if extracting:
            framesArray.append(currentFrame)

    print(len(framesArray))

    footageIndex += 1


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

footageIndex = 1
kernel = np.ones((5, 5), np.uint8)
mouseX, mouseY = 0, 0
framesArray = []
shapeList = []
posList = []
frameIndex = 0
success = True

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, trackbar_onChange)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, trackbar_onChange)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, trackbar_onChange)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, trackbar_onChange)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, trackbar_onChange)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, trackbar_onChange)
prepare_frames()


while True:
    currentFrame = framesArray[0]
    while frameIndex <= len(framesArray):
        currentFrame = framesArray[frameIndex]
        # cv2.waitKey waits for a keypress or its parameter in ms, whichever comes first
        # unless the parameter is 0, then it will wait for the keypress forever
        key = cv2.waitKeyEx(1)

        imgHSV = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

        #  0 179 0 37 0 103 looks promising
        # print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(currentFrame, currentFrame, mask=mask)

        # cv2.imshow("Original",img)
        # cv2.imshow("HSV",imgHSV)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Result", imgResult)

        imgStack = stack_images(0.6, ([currentFrame, imgHSV], [mask, imgResult]))
        cv2.imshow(("dashcam_footage"), imgStack)

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

        # Quit when 'q' is pressed
        if key == ord('i'):
            print(648-649)


cap.release()
cv2.destroyAllWindows()
