import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy
from functools import partial


def setROIPoints():
    global roiPoints
    global roiTopLeft
    global roiTopRight
    global roiBottomLeft
    global roiBottomRight

    # Calculating the corner points for the ROI (The area that is going to be transformed)
    roiTopLeft = [roiHoriz, roiVert - roiHeight]
    roiTopRight = [roiHoriz + roiWidth, roiVert - roiHeight]
    roiBottomLeft = [roiHoriz, roiVert]
    roiBottomRight = [roiHoriz + roiWidth, roiVert]
    roiPoints = np.float32([[roiTopLeft], [roiTopRight], [roiBottomLeft], [roiBottomRight]])


def setTrapezoidPoints():
    global trapezoidPoints
    # Calculating the corner points for the trapezoid (The shape that the ROI will be transformed into)
    trapTopLeft = [0, 0]
    trapTopRight = [frameWidth, 0]
    trapBottomLeft = [trapAng, trapHeight]
    trapBottomRight = [frameWidth - trapAng, trapHeight]
    trapezoidPoints = np.float32([trapTopLeft, trapTopRight, trapBottomLeft, trapBottomRight])


def setPerspective():
    global M
    global invM

    M = cv2.getPerspectiveTransform(roiPoints, trapezoidPoints)  # The transformation matrix
    invM = cv2.getPerspectiveTransform(trapezoidPoints, roiPoints)  # Inverse transformation


def setCroppedPoints():
    global warpedImg
    global croppedImg
    global M
    global invM

    currentFrame = framesArray[frameIndex]
    croppedImgPoints = np.float32([[0, 0], [roiWidth, 0], [0, roiHeight], [roiWidth, roiHeight]])
    croppedImg = currentFrame[roiTopLeft[1]:(roiTopLeft[1] + roiHeight), roiTopLeft[0]:roiTopRight[0]]
    M = cv2.getPerspectiveTransform(croppedImgPoints, trapezoidPoints)  # The transformation matrix
    invM = cv2.getPerspectiveTransform(trapezoidPoints, croppedImgPoints)  # Inverse transformation


def process_frame():
    global framesArray
    global invM
    global rectImg
    global frameWidth
    global warpedImg

    currentFrame = framesArray[frameIndex]
    frameHeight, frameWidth, channels = currentFrame.shape


    # creating a copy of the current frame to add the roi visuals to avoid affecting the original frame
    rectImg = deepcopy(currentFrame)


    warpedImg = cv2.warpPerspective(framesArray[frameIndex], M, (frameWidth, trapHeight))

    #if cropRoi is 1:
        # setCroppedPoints()

    # else:
        # setPerspective()


    # I am using trackbars as a toggle for the rect and the bars
    if showROI is 1:
        cv2.rectangle(rectImg, (roiTopLeft[0], roiTopLeft[1]), (roiBottomRight[0], roiBottomRight[1]), (255, 0, 0), 3)

        cv2.line(rectImg, (int((roiTopRight[0] + roiTopLeft[0]) / 2), 0), (int((roiTopRight[0] + roiTopLeft[0]) / 2), frameHeight), (255, 255, 255), 1)

        #cv2.line(rectImg, (int(frameWidth / 2), 0),
        #         (int(frameWidth / 2), frameHeight), (0, 0, 0), 2)


    if showBars is 1:
        # using the following linear equation I can scale the pixels/meter based on the height of the ROI
        # y = -3.356345*x + 1256.599
        # Where y is the amount of pixels for 7.81 meters and x is the height
        linearMeter = ((-3.356345 * roiHeight) + 1256.599) / 7.81
        barDiff = abs(barTwo - barOne)
        print(barDiff)
        pixelScale = barDiff / 1296
        print('pixel scale', roiWidth * pixelScale)
        realDiff = roiWidth * pixelScale

        meters = round((realDiff / linearMeter), 2)
        print('meters', meters)
        # print(barDiff, meters)
        warningColour = (0, 0, 0)

        # Sets the text's colour based on the meters measured
        if meters < 3:
            # Red
            warningColour = (0, 0, 255)
        elif 4 > meters > 3:
            # Orange
            warningColour = (0, 140, 255)
        elif 6 > meters > 4:
            # Yellow
            warningColour = (0, 255, 255)
        elif meters >= 6:
            # Green
            warningColour = (0, 255, 0)

        # Draw the lines on the warped perspective image
        # Draw the left vertical  bar
        cv2.line(warpedImg, (barOne, 0), (barOne, trapHeight), warningColour, 2)

        # Draw the center horizontal bar
        cv2.line(warpedImg, (barOne, 0), (barTwo, 0), warningColour, 2)

        # Draw the right vertical  bar
        cv2.line(warpedImg, (barTwo, 0), (barTwo, trapHeight), warningColour, 2)

        barOneArr = np.array([[[barOne, 0], [barOne, trapHeight]]], np.float32)
        barTwoArr = np.array([[[barTwo, 0], [barTwo, trapHeight]]], np.float32)


        b1m = cv2.perspectiveTransform(barOneArr, invM)
        b2m = cv2.perspectiveTransform(barTwoArr, invM)



        cv2.line(rectImg, (b1m[0][0][0], b1m[0][0][1]), (b1m[0][1][0], b1m[0][1][1]),
                 warningColour, 2)

        cv2.line(rectImg, (b2m[0][0][0], b2m[0][0][1]), (b2m[0][1][0], b2m[0][1][1]),
                 warningColour, 2)




        if showText:
            barThreeArr = np.array([[[barOne, 0], [barTwo, 0]]], np.float32)
            b3m = cv2.perspectiveTransform(barThreeArr, invM)
            cv2.line(rectImg, (b3m[0][0][0], b3m[0][0][1]), (b3m[0][1][0], b3m[0][1][1]),
                     warningColour, 2)

            displayText = str(meters) + 'm'
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.2
            thickness = 2
            textSize, baseLine = cv2.getTextSize(displayText, font, fontScale, thickness)



            cv2.putText(rectImg, displayText, (int((b3m[0][0][0]+b3m[0][1][0])/2) - (int(textSize[0]/2)),
                                               int((b3m[0][0][1]+b3m[0][1][1])/2) - 10), font,
                        fontScale, warningColour, thickness, cv2.LINE_AA)


    scaledDimRect = getScaledDim(frameWidth, frameHeight, frameScale)
    scaledDimWarped = getScaledDim(frameWidth, trapHeight, frameScale)


    imgInverse = cv2.warpPerspective(warpedImg, invM, (frameWidth, frameHeight))  # Inverse transformation
    scaledImgInverse = cv2.resize(imgInverse, scaledDimRect)
    scaledRectImg = cv2.resize(rectImg, scaledDimRect)
    scaledWarpedImg = cv2.resize(warpedImg, scaledDimWarped)

    cv2.imshow("Dashcam Footage", scaledRectImg)
    cv2.imshow("Birds Eye", scaledWarpedImg)
    # cv2.imshow("Inverse Transform", scaledImgInverse)
    cv2.setMouseCallback("Birds Eye", onMouseBirdsEye)


def getScaledDim(fWidth, fHeight, scale):
    scaledWidth = int(fWidth * scale)
    scaledHeight = int(fHeight * scale)
    return scaledWidth, scaledHeight


def trackbar_onChange(a, var):
    # globals() is a dict of all global variables with the variable names being a dict value
    globals()[var] = a

    process_frame()
    pass


def auto_scale(a):
    newTrapAng = int(a * 1.75)
    oldTrapAng = cv2.getTrackbarPos("Trap Ang", "TrackBars")

    trapAngDiff = int(oldTrapAng - newTrapAng)
    cv2.setTrackbarPos("Trap Ang", "TrackBars", newTrapAng)

    barOneOld = copy(cv2.getTrackbarPos("Bar 1", "Measuring Bars"))
    barTwoOld = copy(cv2.getTrackbarPos("Bar 2", "Measuring Bars"))
    if barOneOld > 647:
        newBarVal = barOneOld + trapAngDiff
    else:
        print(trapAngDiff, barOneOld)
        newBarVal = barOneOld - trapAngDiff

    print(newBarVal)
    cv2.setTrackbarPos("Bar 1", "Measuring Bars", int(newBarVal))
    print(cv2.getTrackbarPos("Bar 1", "Measuring Bars"))

    if barTwoOld > 647:
        newBarVal = barTwoOld + trapAngDiff
    else:
        newBarVal = barTwoOld - trapAngDiff

    cv2.setTrackbarPos("Bar 2", "Measuring Bars", newBarVal)


def test_scale(a):
    oldTrapAng = cv2.getTrackbarPos("Trap Ang", "TrackBars")
    trapAngDiff = int(oldTrapAng - a)

    barOneOld = copy(cv2.getTrackbarPos("Bar 1", "Measuring Bars"))
    barTwoOld = copy(cv2.getTrackbarPos("Bar 2", "Measuring Bars"))
    if barOneOld > 647:
        newBarVal = barOneOld + trapAngDiff
    else:
        newBarVal = barOneOld - trapAngDiff

    print(newBarVal)
    cv2.setTrackbarPos("Bar 1", "Measuring Bars", int(newBarVal))
    print(cv2.getTrackbarPos("Bar 1", "Measuring Bars"))

    if barTwoOld > 647:
        newBarVal = barTwoOld + trapAngDiff
    else:
        newBarVal = barTwoOld - trapAngDiff

    cv2.setTrackbarPos("Bar 2", "Measuring Bars", newBarVal)


def trackbar_onChange_ROI(a, var):
    # globals() is a dict of all global variables with the variable names being a dict value
    globals()[var] = a
    if var is "roiHeight" and autoScale is 1:
        auto_scale(a)

    setROIPoints()
    setPerspective()
    process_frame()

    pass


def trackbar_onChange_trap(a, var):
    # globals() is a dict of all global variables with the variable names being a dict value
    globals()[var] = a
    if var is "trapAng" and autoScale is 1:
        test_scale(a)
    setTrapezoidPoints()
    setPerspective()
    process_frame()
    pass


def trackbar_onChange_frameScale(a):
    global frameScale
    frameScale = a / 100
    process_frame()
    pass


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



def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('col = %d, row = %d' % (x, y))
    return


def onMouseBirdsEye(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 100 = 70/0.7 divide the scaled by the scale to get the original
        p = np.array([[[x/frameScale, y/frameScale]]], np.float32)
        print('col = %d, row = %d' % (x, y))
        m = cv2.perspectiveTransform(p, invM)
        print(m)
        #print('tCol = %d, tRow = %d' % (m[0], m[1]))

        cv2.circle(rectImg, (m[0][0][0], m[0][0][1]), 5, (0, 0, 255), -1)

        scaledWidth = int(1296 * frameScale)
        scaledHeight = int(972 * frameScale)
        scaledDim = (scaledWidth, scaledHeight)
        scaledRectImg = cv2.resize(rectImg, scaledDim)
        cv2.imshow("Dashcam Footage", scaledRectImg)


trapezoidPoints = []
roiPoints = []
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
M = 0
warpedImg = 0
rectImg = 0
cropRoi = 0
roiTopLeft = 0
roiTopRight = 0
roiBottomLeft = 0
roiBottomRight = 0
croppedImg = 0
load_frames(True)
frameHeight, frameWidth, channels = framesArray[frameIndex].shape
#currentFrame = framesArray[frameIndex]


cv2.namedWindow("TrackBars", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Scale", "TrackBars", 80, 100, partial(trackbar_onChange_frameScale))
cv2.createTrackbar("TrapHeight", "TrackBars", 489, 972, partial(trackbar_onChange_trap, var="trapHeight"))
cv2.createTrackbar("Trap Ang", "TrackBars", 301, round(IMAGE_W / 2)-1, partial(trackbar_onChange_trap, var="trapAng"))
cv2.createTrackbar("ROI Height", "TrackBars", 176, 972, partial(trackbar_onChange_ROI, var="roiHeight"))
cv2.createTrackbar("ROI Width", "TrackBars", 1175, 1296, partial(trackbar_onChange_ROI, var="roiWidth"))
cv2.createTrackbar("ROI Horiz", "TrackBars", 111, 1296, partial(trackbar_onChange_ROI, var="roiHoriz"))
cv2.createTrackbar("ROI Vert", "TrackBars", 971, 1296, partial(trackbar_onChange_ROI, var="roiVert"))

cv2.namedWindow("Measuring Bars", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Bar 1", "Measuring Bars", 111, 1296, partial(trackbar_onChange, var="barOne"))
cv2.createTrackbar("Bar 2", "Measuring Bars", 1155, 1296, partial(trackbar_onChange, var="barTwo"))
cv2.createTrackbar("Bar Length", "Measuring Bars", 0, 100, partial(trackbar_onChange, var="barLength"))
cv2.createTrackbar("Auto Scale", "Measuring Bars", 1, 1, partial(trackbar_onChange, var="autoScale"))

cv2.namedWindow("Display Toggles", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Show ROI", "Display Toggles", 1, 1, partial(trackbar_onChange, var="showROI"))
cv2.createTrackbar("Show Bars", "Display Toggles", 1, 1, partial(trackbar_onChange, var="showBars"))
cv2.createTrackbar("Show Text", "Display Toggles", 1, 1, partial(trackbar_onChange, var="showText"))
cv2.createTrackbar("Crop ROI", "Display Toggles", 0, 1, partial(trackbar_onChange, var="cropRoi"))



# Initialise global trackbar variables
trapAng = cv2.getTrackbarPos('Trap Ang', 'TrackBars')
trapHeight = cv2.getTrackbarPos('TrapHeight', 'TrackBars')
roiHeight = cv2.getTrackbarPos('ROI Height', 'TrackBars')
roiWidth = cv2.getTrackbarPos('ROI Width', 'TrackBars')
roiHoriz = cv2.getTrackbarPos('ROI Horiz', 'TrackBars')
roiVert = cv2.getTrackbarPos('ROI Vert', 'TrackBars')
barOne = cv2.getTrackbarPos('Bar 1', 'Measuring Bars')
barTwo = cv2.getTrackbarPos('Bar 2', 'Measuring Bars')
showROI = cv2.getTrackbarPos('Show ROI', 'Display Toggles')
showBars = cv2.getTrackbarPos('Show Bars', 'Display Toggles')
showText = cv2.getTrackbarPos('Show Text', 'Display Toggles')
barLength = cv2.getTrackbarPos('Bar Length', 'TrackBars')
rulerToggle = cv2.getTrackbarPos('Show Ruler', 'TrackBars')
autoScale = cv2.getTrackbarPos('Auto Scale', 'TrackBars')
frameScale = cv2.getTrackbarPos('Scale', 'TrackBars') / 100


# Initialise points and transform matrix
setROIPoints()
setTrapezoidPoints()
setPerspective()


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

    # go to previous frame
    if key == leftArrowCode or key == ord('a'):
        if frameIndex < 1 and videoIndex > 1:
            load_frames(False)
        else:
            frameIndex -= 1

    # Quit when 'q' is pressed
    if key == ord('q'):
        break

    if key == ord('k'):
        break

cv2.destroyAllWindows()
