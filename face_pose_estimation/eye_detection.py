# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         face_landmarks
# Description:
# Author:       huanghongyi
# Date:         2019/1/10
# -------------------------------------------------------------------------------

# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import dlib
import os
from scipy.spatial import distance

base_path = os.path.join(os.path.dirname(__file__), "..")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(base_path + '/model/shape_predictor_68_face_landmarks.dat')

EYE_AR_THRESH = 0.3  # EAR阈值
EYE_AR_CONSEC_FRAMES = 3  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

MOUTH_AR_THRESH = 0.8  # mouth阈值
# 对应特征点的序号
OUT_MOUTH1_START = 49 - 1
OUT_MOUTH1_END = 60 - 1


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def aspect_ratio(q):
    # print(eye)
    q_len = len(q)
    times = int(q_len / 2)
    assert q_len % 2==0

    euclideanSum = 0
    for i in range(1, times):
        f = q[i]
        e = q[q_len - i]
        euclideanSum += distance.euclidean(f, e)

    width = distance.euclidean(q[0], q[times])
    res = euclideanSum / (2.0 * width)
    return res


def eye_aspect_ratio(eye):
    return aspect_ratio(eye)


def mouth_aspect_ratio(mouth):
    return aspect_ratio(mouth)


frame_counter = 0  # 连续帧计数
blink_counter = 0  # 眨眼计数

frame_counter_mouth = 0
open_mouth_counter = 0  # 张嘴

cap = cv2.VideoCapture(0)
while (1):
    ret, img = cap.read()  # 读取视频流的一帧

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
    rects = detector(gray, 0)  # 人脸检测
    for rect in rects:  # 遍历每一个人脸
        print('-' * 20)
        shape = predictor(gray, rect)  # 检测特征点
        points = shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼对应的特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
        leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR
        rightEAR = eye_aspect_ratio(rightEye)  # 计算右眼EAR
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0  # 求左右眼EAR的均值

        leftEyeHull = cv2.convexHull(leftEye)  # 寻找左眼轮廓
        rightEyeHull = cv2.convexHull(rightEye)  # 寻找右眼轮廓
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)  # 绘制左眼轮廓
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)  # 绘制右眼轮廓

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if ear < EYE_AR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0

        mouth = points[OUT_MOUTH1_START:OUT_MOUTH1_END + 1]  # 取出外嘴唇
        mouthEAR = mouth_aspect_ratio(mouth)
        print('mouthEAR = {0}'.format(mouthEAR))
        outMouthHull = cv2.convexHull(mouth)  # 寻找外嘴唇轮廓
        cv2.drawContours(img, [outMouthHull], -1, (0, 255, 0), 1)  # 绘制外嘴唇轮廓

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if mouthEAR < MOUTH_AR_THRESH:
            frame_counter_mouth += 1
        else:
            if frame_counter_mouth >= EYE_AR_CONSEC_FRAMES:
                open_mouth_counter += 1
            frame_counter_mouth = 0

        # 在图像上显示出眨眼次数blink_counter和EAR
        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(img, "OpenMouth:{0}".format(open_mouth_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(img, "mouthEAR:{:.2f}".format(mouthEAR), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
