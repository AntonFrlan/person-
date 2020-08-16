import numpy as np
import cv2 as cv
import svmTraining as svmT

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

train_txt = open('svm.txt', 'r')
train_bool = train_txt.read(1)
train_txt.close()

if train_bool == 'Y' or train_bool == "":
    svmT.svmTrain()

svm = cv.ml.SVM_load('TrainedSVM')

cv.startWindowThread()
cap = cv.VideoCapture('person_014.mp4')

# if True:
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv.imread('person_014.bmp', cv.IMREAD_COLOR)

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('dg', gray)

    # Detect corner
    gray = np.float32(gray)
    gray = cv.goodFeaturesToTrack(gray, 10000, 0.01, 1, useHarrisDetector=True)

    #    i = len(gray)
    #    for x in range(i):
    #        frame[int(gray[x][0][1]), int(gray[x][0][0])] = [0, 0, 255]

    # Display the resulting frame
    cv.imshow('frame', frame)

    if (cv.waitKey(1) & 0xFF == ord('q')) or ret == False:
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
