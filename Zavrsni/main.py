import numpy as np
from imutils.object_detection import non_max_suppression
import cv2 as cv
import imutils
import loadPics

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
    width = int((xb - xa) / 2)
    height = int((yb - ya) / 2)
    if dot is not None:
        xa = dot[0] - width
        ya = dot[1] - height
        xb = dot[0] + width
        yb = dot[1] + height
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
            return [y, x]
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

#------------------------------------------------------------------------------------------------------------------

pics = open('pics.txt', 'r')
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
old_frame = cv.imread(pics.readline())
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

oldDot = np.array([[168, 294]])
oldDot = np.float32(oldDot)
p0 = cv.goodFeaturesToTrack(old_gray, 10000, 0.01, 1, useHarrisDetector=True)  # harris
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
for i in range(1, picNum):
    frame = cv.imread(pics.readline())
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    dot, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, oldDot, None, **lk_params)#DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDIIIIIIIIIIIIIIIOOOOOOOOOOOOOOOOOO
    diff = dot - oldDot
    a,b = dot.ravel()
    frame = cv.circle(frame,(a,b),5,(0, 0, 255),-1)

    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    oldDot = np.float32(dot.copy())
pics.close()
cv.destroyAllWindows()


n = 0
firstFrame = None
pics = open('pics.txt', 'r')
for i in range(picNum):
    if firstFrame is not None:
        oldFrame = frame.copy()
    frame = cv.imread(pics.readline())
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv.absdiff(firstFrame, gray)
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
    for c in cnts:
        # if the contour is too small, ignore it
        if cv.contourArea(c) < 500:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv.boundingRect(c)
        #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        H = harris(thresh)
        centeredRectangle(getDot(frame, H, x, y, x + w, y + h), frame, x, y, x + w, y + h)


    # show the frame and record if the user presses a key
    cv.imshow("Security Feed", frame)
    cv.imshow("Thresh", thresh)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    pause(key)
    n += 1

pics.close()
cv.destroyAllWindows()
