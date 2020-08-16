import numpy as np
import cv2 as cv

def svmTrain():
    train_txt = open('svm.txt', 'w')

    print('Treniranje SVM-a pocinje......')
    # first round
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(0.4)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))

    #data here :(

    train_txt.seek(0)
    train_txt.truncate()
    train_txt.write('Y')
    train_txt.close()
    cv.Algorithm.save(svm, 'TrainedSVM')
    print('....Treniranje SVM-a je gotovo')