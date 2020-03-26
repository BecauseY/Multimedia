
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
from scipy.stats.stats import pearsonr
import time


if __name__ == "__main__":
    print(sys.executable)
    THREADHOLD = 0.99
    # Video path of the source file
    videopath = 'Most_Popular_Programming_Languages_1965_-_2019.mp4'
    # Directory to store the processed frames
    dir = './extract_result/'
    print("target video :" + videopath)
    print("frame save directory: " + dir)
    # load video and compute diff between frames
    cap = cv2.VideoCapture(str(videopath))
    curr_frame = None
    prev_frame = None
    frame_coor = []
    frames = []
    # keyframe_id_set = set()
    start_time = time.time()
    success, frame = cap.read()
    frame_yeartime = frame[560:, 800:, :]
    cv2.imwrite(dir + "1.jpg", frame)
    i = 1
    while (success):
        luv = cv2.cvtColor(frame_yeartime, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        if curr_frame is not None and prev_frame is not None:
            # logic here
            img0 = curr_frame.reshape(curr_frame.size, order='C')  # 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
            img1 = prev_frame.reshape(prev_frame.size, order='C')

            # corr = np.corrcoef(img0, img0)[0, 1]
            corr = pearsonr(img0, img1)[0]
            if corr <= THREADHOLD:
                # keyframe_id_set.add(i-1)
                print(i)
                # rgb = cv2.cvtColor(frame, cv2.COLOR_LUV2BGR)
                name = str(i) + ".jpg"
                cv2.imwrite(dir + name, frame)
        prev_frame = curr_frame
        i = i + 1
        frame_yeartime = frame[560:, 800:, :]
        success, frame = cap.read()

    cap.release()
    end_time = time.time()

    print("耗时：", start_time - end_time)