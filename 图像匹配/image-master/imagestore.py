import cv2
import csv
import numpy as np

def getColorVec(img):

    hei, width, channel=img.shape
    colorVec=[0 for e in range(0, int(pow(COLOR_DEGREE, 3)))]
    i=0
    while(i<hei):
        j=0
        while(j<width):
            pixel=img[i][j]
            grade=getPixelGrade(pixel)
            index=grade[0]*COLOR_DEGREE*COLOR_DEGREE+grade[1]*COLOR_DEGREE+grade[2]
            colorVec[index]+=1
            j+=1
        i+=1
    return colorVec


def getPixelGrade(pixel):
    grade=[]
    base=int(256/COLOR_DEGREE)+1
    for one in np.array(pixel):
        grade.append(int(one/base))
    return grade

if __name__ == '__main__':
    COLOR_DEGREE = 8
    for bigi in range(0,9908):
        img = cv2.imread('../image/' + str(bigi) + '.jpg')
        hist=getColorVec(img)
        with open('test8.csv', 'a+',newline='')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(hist)





