import os

def loadImage(pathImage, pathTxt):
    imageTxt = open(pathTxt, 'w')
    picNum = 0
    for files in os.listdir(pathImage):
        if os.path.isfile(os.path.join(pathImage, files)):
            imageTxt.write(pathImage + '\\' + files)
            imageTxt.write('\0\n')
            picNum += 1
    imageTxt.close()
    return picNum


def pics():
    return loadImage('C:\\Users\\Anton\\Desktop\\Zavrsni\\Zavrsni\\pics', 'pics.txt')
