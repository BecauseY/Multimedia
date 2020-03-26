import cv2
import numpy as np
import csv
import re
from math import sqrt
import matplotlib.pyplot as plt

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


def Bdistance(l1, l2):
    if(len(l1)!=len(l2)):
        raise RuntimeError("计算巴氏距离时，引入长度不相等的向量")
    s1=sum(l1)
    s2=sum(l2)
    BD=0
    for ind in range(0, len(l1)):
        BD+=sqrt((l1[ind]/s1)*(l2[ind]/s2))
    return BD

if __name__ == '__main__':
    import operator
    imgpath='../image/1732.jpg'
    img = cv2.imread(imgpath)
    imgnum=(int(re.sub("\D", "", imgpath)))//100
    COLOR_DEGREE=8
    hist1 = getColorVec(img)
    gray_dict = {}
    jpgi=0
    with open('test8.csv')as f:
        f_csv = csv.reader(f)
        for csvrow in f_csv:
            hist2 = [int(row) for row in csvrow]  # 将数据从string形式转换为float形式
            gray_dict[str(jpgi) + '.jpg'] = Bdistance(hist1, hist2)
            jpgi=jpgi+1

    gray_dict = sorted(gray_dict.items(),key=operator.itemgetter(1))

    dic100=gray_dict[-100:]
    totalnum=0
    for i in range(0,len(dic100)):
        num=int(re.sub("\D", "", dic100[i][0]))
        print(num)
        if num>=imgnum*100 and num<=imgnum*100+200:
            totalnum=totalnum+1
    print(dic100)
    print('查找到的图象个数：',totalnum)
    print('查全率：',totalnum/100.0)


