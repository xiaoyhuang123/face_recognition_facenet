# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         FacePoseEstimator
# Description:  
# Author:       huanghongyi
# Date:         2019/1/9
# -------------------------------------------------------------------------------

import cv2
import numpy as np
import dlib
import time
import math
import os
from utils import Logger

logger = Logger(__name__)

class FacePoseEstimator:

    def __init__(self, model):
        self.detector = dlib.get_frontal_face_detector()
        f_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + model
        self.predictor = dlib.shape_predictor(f_dir)
        self.POINTS_NUM_LANDMARK = 68

    # 获取最大的人脸
    def _largest_face(self, dets):
        if len(dets) == 1:
            return 0

        face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]

        largest_area = face_areas[0]
        largest_index = 0
        for index in range(1, len(dets)):
            if face_areas[index] > largest_area:
                largest_index = index
                largest_area = face_areas[index]

        print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

        return largest_index

    def processImage(self, img):
        size = img.shape
        if size[0] > 700:
            h = size[0] / 3
            w = size[1] / 3
            img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
            size = img.shape
        return size, img

    # 从dlib的检测结果抽取姿态估计需要的点坐标
    def get_image_points_from_landmark_shape(self, landmark_shape):
        if landmark_shape.num_parts != self.POINTS_NUM_LANDMARK:
            print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
            return -1, None

        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            (landmark_shape.part(30).x, landmark_shape.part(30).y),  # Nose tip
            (landmark_shape.part(8).x, landmark_shape.part(8).y),  # Chin
            (landmark_shape.part(36).x, landmark_shape.part(36).y),  # Left eye left corner
            (landmark_shape.part(45).x, landmark_shape.part(45).y),  # Right eye right corne
            (landmark_shape.part(48).x, landmark_shape.part(48).y),  # Left Mouth corner
            (landmark_shape.part(54).x, landmark_shape.part(54).y)  # Right mouth corner
        ], dtype="double")

        return 0, image_points

    # 用dlib检测关键点，返回姿态估计需要的几个点坐标
    def get_image_points(self, img):
        # gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
        dets = self.detector(img, 0)

        if 0 == len(dets):
            print("ERROR: found no face")
            return -1, None
        largest_index = self._largest_face(dets)
        face_rectangle = dets[largest_index]

        landmark_shape = self.predictor(img, face_rectangle)

        return self.get_image_points_from_landmark_shape(landmark_shape)

    # 获取旋转向量和平移向量
    def get_pose_estimation(self, img_size, image_points):
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        # Camera internals

        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        print("Camera Matrix :{}".format(camera_matrix))
        logger.info("Camera Matrix :{}".format(camera_matrix))

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        print("Rotation Vector:\n {}".format(rotation_vector))
        print("Translation Vector:\n {}".format(translation_vector))
        logger.info("Rotation Vector:\n {}".format(rotation_vector))
        logger.info("Translation Vector:\n {}".format(translation_vector))
        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

    # 从旋转向量转换为欧拉角
    def get_euler_angle(self, rotation_vector):
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)

        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta

        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        print('t0:{}, t1:{}'.format(t0, t1))
        pitch = math.atan2(t0, t1)
        # 但是我们发现第pitch坐标抖动很厉害，而且是在正负之间震荡，
        # 但是在一定范围内是随点头的幅度呈近似线性关系，所以考虑是坐标角度变换的问题，在减去180度发现是从0度直接到-360，所以是互补的问题：
        if(pitch<=-180):
            pitch = pitch+180

        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)

        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)

        print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        # 单位转换：将弧度转换为度
        Y = int((pitch / math.pi) * 180)
        X = int((yaw / math.pi) * 180)
        Z = int((roll / math.pi) * 180)

        return 0, Y, X, Z

    def doFaceEstimater(self, im):
        size, im = self.processImage(im)

        ret, image_points = self.get_image_points(im)
        if ret != 0:
            print('get_image_points failed')
            return im,None,None,None

        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = self.get_pose_estimation(size,
                                                                                                        image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return im,None,None,None
        # used_time = time.time() - start_time
        # print("used_time:{} sec".format(round(used_time, 3)))

        ret, pitch, yaw, roll = self.get_euler_angle(rotation_vector)
        euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        print(euler_angle_str)

        # Yaw:摇头
        # 左正右负
        #
        # Pitch:点头
        # 上负下正
        #
        # Roll:摆头（歪头）左负
        # 右正

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.putText(im, str(rotation_vector), (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(im, euler_angle_str, (0, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        cv2.line(im, p1, p2, (255, 0, 0), 2)
        return im,pitch, yaw, roll



if __name__ == '__main__':

    face_estimator_model = "/model/shape_predictor_68_face_landmarks.dat"
    fpe = FacePoseEstimator(face_estimator_model)

    # rtsp://admin:ts123456@10.20.21.240:554
    cap = cv2.VideoCapture(0)

    frame_counter = 0  # 连续帧计数
    shake_counter = 0  # 眨眼计数

    while (cap.isOpened()):
        start_time = time.time()

        # Read Image
        ret, im = cap.read()
        if ret != True:
            print('read frame failed')
            continue
        size = im.shape

        im,pitch, yaw, roll = fpe.doFaceEstimater(im)

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if(yaw is not None):
            if yaw < 0.3:
                frame_counter += 1
            else:
                if frame_counter > 0.3:
                    shake_counter += 1
                frame_counter = 0

        cv2.putText(im, "shakeHead:{0}".format(shake_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Output", im)

        cv2.waitKey(1)
