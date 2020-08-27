import numpy as np
from imutils.object_detection import non_max_suppression
import cv2 as cv
import imutils
import loadPics
import time

def harris(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # prebacivanje u sivu jer moram
    gray = np.float32(gray)  # mora biti tog tipa
    return cv.goodFeaturesToTrack(gray, 10000, 0.01, 1, useHarrisDetector=True)  # harris


def pause(key):#pauziranje i pregledavanje rezultata
    if key == ord("w"):
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord("e"):
                break


def centeredRectangle(dot, frame, xa, ya, xb, yb):#crtanje pravokutnika s centroidom u centru
    width = (xb - xa) / 2
    height = (yb - ya) / 2
    if dot is not None:
        xa = int(dot[0] - width)
        ya = int(dot[1] - height)
        xb = int(dot[0] + width)
        yb = int(dot[1] + height)
        cv.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 2)


def getDot(src, H, xa=0, ya=0, xb=99999, yb=99999):#racunanje centroida u meti pracenja pomocu karakteristcnih tocaka
    if H is not None:
        g = 0
        x = 0
        y = 0
        for i in H:
            a = int(i[0][1])
            b = int(i[0][0])
            thresh[a][b] = (0, 0, 255)
            if xa < b < xb and ya < a < yb:
                x += int(i[0][1])
                y += int(i[0][0])
                g += 1
        if g > 0:
            x = int(x / g)
            y = int(y / g)
            cv.circle(src, (y, x), 5, (0, 0, 255), -1)
            return np.array([y, x])
        return None

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


picNum = loadPics.pics()
print(picNum)

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cv.startWindowThread()
#cap = cv.VideoCapture('person_014.mp4')

# if True:
#while (True):
    # Capture frame-by-frame
    #ret, frame = cap.read()
pics = open('pics.txt', 'r')
newFrame = cv.imread(pics.readline())
newH = harris(newFrame)
firstFrame = newFrame.copy()
for i in range(1, picNum):
    oldFrame = newFrame.copy()
    oldH = newH.copy()
    frame = cv.imread(pics.readline())
    # Our operations on the frame come here
    frame = imutils.resize(frame, width=min(800, frame.shape[1]))
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(4, 4), scale=1.05)

    #for (x, y, w, h) in rects:
    #    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Detect corner
    #gray = np.float32(gray)
    #gray = cv.goodFeaturesToTrack(gray, 10000, 0.01, 1, useHarrisDetector=True)

    #    i = len(gray)
    #    for x in range(i):
    #        frame[int(gray[x][0][1]), int(gray[x][0][0])] = [0, 0, 255]

    # Display the resulting frame
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
#cap.release()
cv.destroyAllWindows()
pics.close()

#--------------------------------------------------------------------------------------------------------------------------

pics = open('pics.txt', 'r')
frame = cv.imread(pics.readline())
oldGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(oldGray, 10000, 0.01, 1, useHarrisDetector=True)  # harris

n = 11
firstFrame = cv.GaussianBlur(oldGray, (21, 21), 0)
for i in range(1, picNum):
    frame = cv.imread(pics.readline())
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gaus = cv.GaussianBlur(gray, (21, 21), 0)


    # compute the absolute difference between the current frame and
    # first frame
    if n == 1:
        oldDots = dots.copy()
        dots, st, err = cv.calcOpticalFlowPyrLK(oldGray, gray, oldDots, None, winSize=(15, 15),
                                                maxLevel=2,
                                                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        diff = dots - oldDots

    if(n == 10):
        n = 0
        frameDelta = cv.absdiff(firstFrame, gaus)
        thresh = cv.threshold(frameDelta, 50, 255, cv.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv.dilate(thresh, None, iterations=2)
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        T = cv.bitwise_and(oldFrame, oldFrame, mask=thresh)
        cv.imshow("oldT", T)
        thresh = cv.bitwise_and(frame, frame, mask=thresh)
        # loop over the contours
        size = 0
        for c in cnts:
            size += 1
        if size == 0:
            n = 10
            continue
        dots = np.zeros((size, 2), dtype=np.float32)
        j = -1
        for c in cnts:
            j += 1
            # if the contour is too small, ignore it
            if cv.contourArea(c) < 500:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv.boundingRect(c)
            #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            H = harris(thresh)
            dots[j] = getDot(frame, H, x, y, x + w, y + h)
            if dots[j] is not None:
                centeredRectangle(dots[j], frame, x, y, x + w, y + h)
    elif n == 11:
        n = 10
        oldGray = gray.copy()
        continue
    else:

        for j in range(size):
            if dots[j] is not None:
                dots[j] += diff[j]
                a, b = dots[j].ravel()
                frame = cv.circle(frame, (a, b), 5, (0, 0, 255), -1)


    # show the frame and record if the user presses a key
    cv.imshow("Security Feed", frame)
    cv.imshow("Thresh", thresh)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    pause(key)

    oldGray = gray.copy()
    n += 1
    time.sleep(1)

pics.close()
cv.destroyAllWindows()
