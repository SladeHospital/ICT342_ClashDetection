import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy


def process_frame():
    global framesArray
    global currentFrame
    global invM
    global rectImg

    trapTopWidth = cv2.getTrackbarPos('Trap Top', 'TrackBars')
    trapAng = cv2.getTrackbarPos('Trap Ang', 'TrackBars')
    trapHeight = cv2.getTrackbarPos('Win Height', 'TrackBars')
    roiHeight = cv2.getTrackbarPos('ROI Height', 'TrackBars')
    roiWidth = cv2.getTrackbarPos('ROI Width', 'TrackBars')
    roiHoriz = cv2.getTrackbarPos('ROI Horiz', 'TrackBars')
    roiVert = cv2.getTrackbarPos('ROI Vert', 'TrackBars')
    barOne = cv2.getTrackbarPos('Bar 1', 'TrackBars')
    barTwo = cv2.getTrackbarPos('Bar 2', 'TrackBars')
    showRect = cv2.getTrackbarPos('Show Rect', 'TrackBars')
    showBars = cv2.getTrackbarPos('Show Bars', 'TrackBars')
    barLength = cv2.getTrackbarPos('Bar Length', 'TrackBars')
    rulerToggle = cv2.getTrackbarPos('Show Ruler', 'TrackBars')

    currentFrame = framesArray[frameIndex]
    frameHeight, frameWidth, channels = currentFrame.shape

    # Calculating the corner points for the ROI (The area that is going to be transformed)
    roiTopLeft = [roiHoriz, roiVert - roiHeight]
    roiTopRight = [roiHoriz + roiWidth, roiVert - roiHeight]
    roiBottomLeft = [roiHoriz, roiVert]
    roiBottomRight = [roiHoriz + roiWidth, roiVert]
    roiPoints = np.float32([[roiTopLeft], [roiTopRight], [roiBottomLeft], [roiBottomRight]])

    # Calculating the corner points for the trapezoid (The shape that the ROI will be transformed into)
    trapTopLeft = [0, 0]
    trapTopRight = [frameWidth, 0]
    trapBottomLeft = [trapAng, trapHeight]
    trapBottomRight = [frameWidth - trapAng, trapHeight]
    trapezoidPoints = np.float32([trapTopLeft, trapTopRight, trapBottomLeft, trapBottomRight])

    # The transformation matrix
    matrix = cv2.getPerspectiveTransform(roiPoints, trapezoidPoints)
    # Inverse transformation
    matrixInv = cv2.getPerspectiveTransform(trapezoidPoints, roiPoints)

    # The resulting image after the matrix was applied to the ROI
    warpedImg = cv2.warpPerspective(currentFrame, matrix, (frameWidth, trapHeight))

    # creating a copy of the current frame to add the roi visuals to avoid affecting the original frame
    rectImg = deepcopy(currentFrame)


    # I am using trackbars as a toggle for the rect and the bars
    if showRect is 1:
        cv2.rectangle(rectImg, (roiTopLeft[0], roiTopLeft[1]), (roiBottomRight[0], roiBottomRight[1]), (0, 255, 255), 3)



    if showBars is 1:

        cv2.line(warpedImg, (barOne, 0), (barOne, trapHeight),
                 (0, 255, 0), 2)

        cv2.line(warpedImg, (barTwo, 0), (barTwo, trapHeight), (0, 255, 0), 2)
        # 3M Warning Bars


        barOneArr = np.array([[[barOne, 0], [barOne, trapHeight]]], np.float32)
        barTwoArr = np.array([[[barTwo, 0], [barTwo, trapHeight]]], np.float32)

        b1m = cv2.perspectiveTransform(barOneArr, matrixInv)
        b2m = cv2.perspectiveTransform(barTwoArr, matrixInv)


        cv2.line(rectImg, (b1m[0][0][0], b1m[0][0][1]), (b1m[0][1][0], b1m[0][1][1]),
                 (0, 255, 0), 2)

        cv2.line(rectImg, (b2m[0][0][0], b2m[0][0][1]), (b2m[0][1][0], b2m[0][1][1]),
                 (0, 255, 0), 2)



    scale = 0.7
    scaledDimRect = getScaledDim(frameWidth, frameHeight, scale)
    scaledDimWarped = getScaledDim(frameWidth, trapHeight, scale)

    imgInverse = cv2.warpPerspective(warpedImg, matrixInv, (frameWidth, frameHeight))  # Inverse transformation
    scaledImgInverse = cv2.resize(imgInverse, scaledDimRect)
    scaledRectImg = cv2.resize(rectImg, scaledDimRect)
    scaledWarpedImg = cv2.resize(warpedImg, scaledDimWarped)

    cv2.imshow("Dashcam Footage", scaledRectImg)
    cv2.imshow("Birds Eye", scaledWarpedImg)
    # cv2.imshow("Inverse Transform", scaledImgInverse)
    cv2.setMouseCallback("Birds Eye", onMouseBirdsEye)


def getScaledDim(frameWidth, frameHeight, scale):
    scaledWidth = int(frameWidth * scale)
    scaledHeight = int(frameHeight * scale)
    return scaledWidth, scaledHeight


def trackbar_onChange(a):
    process_frame()
    pass


def convert_point(transformMatrix, inversePoint=None, inputPoint=None):
    if inversePoint is None and inputPoint is not None:
        return transformMatrix * inputPoint
    if inputPoint is None and inversePoint is not None:
        return inversePoint / transformMatrix


def load_frames(nextVideo):
    global frameIndex
    global videoIndex

    framesArray.clear()

    if nextVideo:
        videoIndex += 1
        frameIndex = 0
    else:
        videoIndex -= 1

    cap = cv2.VideoCapture(footageDir + "dashcam_footage_%d.avi" % videoIndex)
    extracting = True
    while extracting:
        extracting, currFrame = cap.read()
        if extracting:
            framesArray.append(currFrame)

    if nextVideo is False:
        frameIndex = len(framesArray) - 1

    print(len(framesArray))


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


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('col = %d, row = %d' % (x, y))
    return


def onMouseBirdsEye(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = 0.7
        # 100 = 70/0.7 divide the scaled by the scale to get the original
        p = np.array([[[x/scale, y/scale]]], np.float32)
        print('col = %d, row = %d' % (x, y))
        m = cv2.perspectiveTransform(p, invM)
        print(m)
        #print('tCol = %d, tRow = %d' % (m[0], m[1]))

        cv2.circle(rectImg, (m[0][0][0], m[0][0][1]), 5, (0, 0, 255), -1)

        scaledWidth = int(1296 * scale)
        scaledHeight = int(972 * scale)
        scaledDim = (scaledWidth, scaledHeight)
        scaledRectImg = cv2.resize(rectImg, scaledDim)
        cv2.imshow("Dashcam Footage", scaledRectImg)



footageDir = "ClashDetectionMaterial/"
rightArrowCode = 2555904
leftArrowCode = 2424832
videoIndex = 0  # the index for the current video
kernel = np.ones((5, 5), np.uint8)
mouseX, mouseY = 0, 0
framesArray = []
frameIndex = 0  # the index of the current frame from the stored frames
success = True
IMAGE_H = 550
IMAGE_W = 1296
invM = 0
rectImg = 0

cv2.namedWindow("TrackBars", cv2.WINDOW_AUTOSIZE)
#cv2.resizeWindow("TrackBars", 600, 390)
cv2.createTrackbar("Trap Top", "TrackBars", 648, round(IMAGE_W / 2), trackbar_onChange)
cv2.createTrackbar("Win Height", "TrackBars", 489, 972, trackbar_onChange)
cv2.createTrackbar("Trap Ang", "TrackBars", 301, round(IMAGE_W / 2)-1, trackbar_onChange)
cv2.createTrackbar("ROI Height", "TrackBars", 176, 972, trackbar_onChange)
cv2.createTrackbar("ROI Width", "TrackBars", 878, 1296, trackbar_onChange)
cv2.createTrackbar("ROI Horiz", "TrackBars", 316, 1296, trackbar_onChange)
cv2.createTrackbar("ROI Vert", "TrackBars", 971, 1296, trackbar_onChange)
cv2.createTrackbar("Bar 1", "TrackBars", 151, 1296, trackbar_onChange)
cv2.createTrackbar("Bar 2", "TrackBars", 1155, 1296, trackbar_onChange)
cv2.createTrackbar("Show Rect", "TrackBars", 1, 1, trackbar_onChange)
cv2.createTrackbar("Show Bars", "TrackBars", 1, 1, trackbar_onChange)
cv2.createTrackbar("Bar Length", "TrackBars", 0, 100, trackbar_onChange)
cv2.createTrackbar("Show Ruler", "TrackBars", 0, 1, trackbar_onChange)


load_frames(True)


currentFrame = framesArray[0]
cv2.imshow("Dashcam Footage", currentFrame)
cv2.setMouseCallback('Dashcam Footage', onMouse)


while frameIndex <= len(framesArray):
    process_frame()

    # cv2.waitKey waits for a keypress or its parameter in ms, whichever comes first
    # unless the parameter is 0, then it will wait for the keypress forever
    key = cv2.waitKeyEx(0)

    # go to next frame
    if key == rightArrowCode or key == ord('d'):
        if frameIndex >= len(framesArray)-1:
            load_frames(True)
        else:
            frameIndex += 1
            print(frameIndex)

    # go to previous frame
    if key == leftArrowCode or key == ord('a'):
        if frameIndex < 1 and videoIndex > 1:
            print(videoIndex)
            load_frames(False)
        else:
            frameIndex -= 1

    # Quit when 'q' is pressed
    if key == ord('q'):
        break

    if key == ord('k'):
        print(currentFrame.shape)


cv2.destroyAllWindows()
