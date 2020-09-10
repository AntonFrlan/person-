import numpy as np
import cv2 as cv

def rectCorners(c):
    xa = int(c.dot[0][0] - c.width)
    ya = int(c.dot[0][1] - c.height)
    xb = int(c.dot[0][0] + c.width)
    yb = int(c.dot[0][1] + c.height)
    return [xa, ya, xb, yb]

def thresholdCalc(firstFrame, gaus):#background subtraction
    t = cv.absdiff(firstFrame, gaus)
    t = cv.threshold(t, 45, 255, cv.THRESH_BINARY)[1]
    return cv.dilate(t, None, iterations=2)


def harris(img):#racunanje harrisovih tocaka
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    return cv.goodFeaturesToTrack(gray, 100000, 0.01, 1, useHarrisDetector=True)


def centeredRectangle(frame, c):#crtanje pravokutnika s centroidom u centru
    if c.dot is not None:
        x = rectCorners(c)
        cv.circle(frame, (c.dot[0][0], c.dot[0][1]), 5, (0, 0, 255), -1)
        cv.rectangle(frame, (x[0], x[1]), (x[2], x[3]), (0, 255, 0), 2)


def getDot(H, xa, ya, xb, yb):#racunanje centroida u meti
    if H is not None:         #pracenja pomocu karakteristcnih tocaka
        g = 0
        x = 0
        y = 0
        for i in H:
            a = int(i[0][1])
            b = int(i[0][0])
            if xa < b < xb and ya < a < yb:
                x += int(i[0][1])
                y += int(i[0][0])
                g += 1
        if g > 0:
            x = int(x / g)
            y = int(y / g)
            return np.float32(np.array([[y, x]]))
        return None
