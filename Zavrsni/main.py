import numpy as np
from imutils.object_detection import non_max_suppression
import cv2 as cv
import imutils
import time
import loadPics
import dotCalculation as d
import centroid
import keys

picNum = loadPics.pics()
if picNum < 1:
    print("No pictures in folder.")
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
#frame = imutils.resize(frame, width=min(800, frame.shape[1]))

pics = open('pics.txt', 'r')

frame = cv.imread(pics.readline())
oldGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
firstFrame = cv.GaussianBlur(oldGray, (21, 21), 0)

HEIGHT, WIDTH, _ = frame.shape
if WIDTH < 101 or HEIGHT < 101:
    print("Frame is too small.")
cen = []#centroids
n = 10
for i in range(1, picNum):
    frame = cv.imread(pics.readline())

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gaus = cv.GaussianBlur(gray, (21, 21), 0)
    thresh = d.thresholdCalc(firstFrame, gaus)
    mask = cv.bitwise_and(frame, frame, mask=thresh)
    H = d.harris(mask)

    if n == 10:
        n = 0

        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                                padding=(4, 4), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (x, y, w, h) in pick:
            notF = True

            for c in cen:
                if c.check(x, y, w, h):
                    c.update(d.getDot(H, x, y, w, h), x, y, w, h)
                    c.disReset()
                    if c.dot is not None:
                        d.centeredRectangle(frame, c)
                        notF = False
                        break
            if notF:
                j = d.getDot(H, x, y, w, h)
                if j is not None:
                    cen.append(centroid.Centroid(j, int((w - x) / 2), int((h - y) / 2)))
                    d.centeredRectangle(frame, cen[-1])
        for c in cen:
            if not c.found:
                c.update(d.getDot(H, c.dot[0][0] - c.width, c.dot[0][1] - c.height, c.dot[0][0] + c.width, c.dot[0][1] + c.height), c.dot[0][0] - c.width, c.dot[0][1] - c.height, c.dot[0][0] + c.width, c.dot[0][1] + c.height)
                d.centeredRectangle(frame, c)
    else:
        for c in cen:
            c.old()
            if c.oldDot is not None:
                c.dot, st, err = cv.calcOpticalFlowPyrLK(oldGray, gray, c.oldDot, None, winSize=(15, 15),
                                                         maxLevel=2,
                                                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

                if c.dot is None or c.errorLK():
                    c.dot = c.oldDot + c.speed
                j = c.center(d.rectCorners(c))
                j = d.getDot(H, j[0], j[1], j[2], j[3])
                if j is not None:
                    c.dot = j
                d.centeredRectangle(frame, c)
                c.changeFound()

    cv.imshow("Security Feed", frame)
    cv.imshow("Thresh", mask)

    if keys.check():
        break

    j = 0
    for c in cen:
        if c.nestanite(WIDTH, HEIGHT):
            del cen[j]
        elif not c.found:
            keys.pause(ord("w"))
        else:
            c.changeFound()
        j += 1
    oldGray = gray.copy()
    n += 1
    time.sleep(0.1)
pics.close()
cv.destroyAllWindows()
