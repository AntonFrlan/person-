import cv2 as cv

def pause(key):#pauziranje i pregledavanje
    if key == ord("w"):# rezultata
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord("e"):
                break
def check():
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        return True
    pause(key)
    return False
