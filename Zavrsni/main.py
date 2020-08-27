import numpy as np
from imutils.object_detection import non_max_suppression
import cv2 as cv
import imutils
import loadPics
import dotCalculation as d
import centroid
import keys

picNum = loadPics.pics()

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
#frame = imutils.resize(frame, width=min(800, frame.shape[1]))

pics = open('pics.txt', 'r')
novo = False

frame = cv.imread(pics.readline())
oldGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
firstFrame = cv.GaussianBlur(oldGray, (21, 21), 0)

HEIGHT, WIDTH, _ = frame.shape
print(WIDTH, HEIGHT)
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
            novo = True
            notF = True

            for c in cen:
                print('HOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOG')
                print(c.dot, 'y is this shit geh', c.width, c.height, c.found)
                print(x, y, w, h)
                if c.check(x, y, w, h):
                    print('usao sam <3 :,)')
                    print(d.getDot(H, x, y, w, h))
                    print(":(")
                    c.update(d.getDot(H, x, y, w, h), x, y, w, h)
                    if c.dot is not None:
                        d.centeredRectangle(frame, c)
                        notF = False
                        break
                    else:
                        print('ODI U TRI PICKE MILE MATERINE SMECE JEDNO SMRDLJIVO LAZOVSKO POKVARENO')
                print('HOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOG')
            for c in cen:#IZVADI OVO VAN!!!!!!!!!! ULAZI KADA NE BI TREBALO PA CRTA ONO STO JE MOZDA U SLJEDECEM KVADRATU, TUGICA!!!!!!!!!!!!!!!!!!!
                    c.update(d.getDot(H, c.dot[0][0] - c.width, c.dot[0][1] - c.height, c.dot[0][0] + c.width, c.dot[0][1] + c.height), c.dot[0][0] - c.width, c.dot[0][1] - c.height, c.dot[0][0] + c.width, c.dot[0][1] + c.height)
                    d.centeredRectangle(frame, c)
            if notF:
                j = d.getDot(H, x, y, w, h)
                if j is not None:
                    cen.append(centroid.Centroid(j, int((w - x) / 2), int((h - y) / 2)))
                    d.centeredRectangle(frame, cen[-1])
    else:
        for c in cen:
            c.old()
            if c.oldDot is not None:
                a = np.float32(np.array([[150, 160]]))
                c.dot, st, err = cv.calcOpticalFlowPyrLK(oldGray, gray, c.oldDot, None, winSize=(15, 15),
                                                        maxLevel=2,
                                                        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
                print('OPTICKI TOK......................................................................................')
                print(c.dot)
                if c.dot is None:
                    print('uso sam')
                    c.dot = c.oldDot + c.speed
                print(c.dot)
                print('OPTICKI TOK......................................................................................')
                j = c.center(d.rectCorners(c))
                c.dot = d.getDot(H, j[0], j[1], j[2], j[3])
                d.centeredRectangle(frame, c)

    cv.imshow("Security Feed", frame)
    cv.imshow("Thresh", mask)
    if novo:
        novo = False
        keys.pause(ord("w"))

    if keys.check():
        break

    j = 0
    for c in cen:
        if c.nestanite(WIDTH, HEIGHT):
            print("NESTANITE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", c.dot, c.dis)
            del cen[j]
        else:
            c.notFound()
            print(c.dot)
        j += 1
        print()
    oldGray = gray.copy()
    n += 1

pics.close()
cv.destroyAllWindows()
