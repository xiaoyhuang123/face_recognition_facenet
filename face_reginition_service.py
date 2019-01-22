# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         face_recognition
# Description:
# Author:       huanghongyi
# Date:         2018/12/17
# -------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import cv2
import copy
import pymysql
import config
from face_pose_estimation.FacePoseEstimator import FacePoseEstimator
from InputObject import DetectObj
from sklearn.neighbors import KNeighborsClassifier
from utils import knn_clf_for_face_reginition,softmax_label,to_rgb,Logger
logger = Logger(__name__)

import logging

logger = logging.getLogger(__name__)

from keras.models import model_from_json
# import win32com.client
# speaker = win32com.client.Dispatch("SAPI.SpVoice")

class face_reginition:

    def __init__(self, dburl, username, password, dbname, model, path, emo_model, estimator_model,
                 detect_multiple_faces=True):
        print("face_reginition init start")
        # 运行模式  0：展示模式  1：人脸检测模式  2：手势识别模式
        self.running_mode = 1

        # 数据库设置相关
        self.dburl = dburl
        self.username = username
        self.password = password
        self.dbname = dbname

        # 预训练好模型
        self.model = model

        # 候选人脸路径设置
        self.path = path
        self.imgtype = {'jpg', 'png'}

        # 人脸检测参数
        self.detect_multiple_faces = detect_multiple_faces
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.image_size = 160
        self.margin = 44

        # 识别准确率阈值,如果距离超过阈值，则认为不匹配，否则匹配成功
        self.min_face_distance = 1

        self.emo_model = emo_model
        self.face_reginition_mode = 1
        self.emotion_reginition_mode = 0
        if (self.emotion_reginition_mode == 1):
            # 加载表情识别模型
            self.load_emotion_model()

        self.fpe = FacePoseEstimator(estimator_model)

        self.fname_list, self.face_encode_list = self.query_face_feature_from_db()
        # self.knn = KNeighborsClassifier(n_neighbors=1)
        # self.knn.fit(self.face_encode_list, self.fname_list)

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        self.graph1 = tf.Graph()
        self.sess1 = tf.Session(graph=self.graph1)
        ####################################################
        with self.sess.as_default():
            with self.graph.as_default():
                # Load the model
                print("face_reginition init load facenet model")
                facenet.load_model(self.model)
                # Get input and output tensors
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                self.pnet, self.rnet, self.onet = self.load_pronet(self.sess)
        print("face_reginition init end")


    def set_runing_mode(self, runing_mode):
        face_reginition_mode = 1
        emotion_reginition_mode = 1
        if len(runing_mode) == 2:
            face_reginition_mode = int(runing_mode[0])
            emotion_reginition_mode = int(runing_mode[1])
        self.face_reginition_mode = face_reginition_mode
        self.emotion_reginition_mode = emotion_reginition_mode

    def load_emotion_model(self):
        # 加载第一个模型
        with self.sess1.as_default():
            with self.sess.graph.as_default():
                json_file = open(self.emo_model, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                print("加载keras模型成功")

                emotion_model = model_from_json(loaded_model_json)
                # load weights into new model
                emotion_model.load_weights('hhy_emo/model.h5')
                print("加载权重成功")
                self.emotion_model = emotion_model

    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    # 获取候选人集合
    def get_candidate_person_pics(self):
        image_files = []
        for person_file in os.listdir(self.path):
            if (not os.path.isdir(path + "/" + person_file)):
                continue
            for pic in os.listdir(path + "/" + person_file):
                split_array = os.path.splitext(pic)
                ty = str.lower(split_array[1])
                if ty == '.png' or ty == '.jpg':
                    image_files.append(person_file + "/" + pic)
        print("--------------image_files------------", image_files)
        logger.info("--------------image_files------------", image_files)
        return image_files

    # load pnet/rnet/onet
    def load_pronet(self,sess, gpu_memory_fraction=1.0):
        print('Creating networks and loading parameters')
        logger.info('Creating networks and loading parameters')
        with sess.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
        return pnet, rnet, onet

    # openCV capture setting
    def get_capture_cv2(self):
        cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)  # 宽度
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)  # 高度
        # cap.set(cv2.CAP_PROP_FPS, 50)  # 帧数
        return cap

    # 获取候选人脸特征数据
    def query_face_feature_from_db(self):
        fname_list = []
        face_encode_list = []

        # 打开数据库连接
        db = pymysql.connect(self.dburl, self.username, self.password, self.dbname, charset='utf8')

        # 使用cursor()方法获取操作游标
        cursor = db.cursor()

        # SQL 查询语句
        sql = "SELECT f_name,f_encode,f_file_name FROM face_data"
        try:
            # 执行SQL语句
            cursor.execute(sql)
            # 获取所有记录列表
            results = cursor.fetchall()
            for row in results:
                fname = row[0]
                face_encode_str = row[1]

                face_encode = face_encode_str.split('#')
                face_encode = list(map(float, face_encode))

                face_file_path = row[2]
                fname_list.append(fname)
                face_encode_list.append(face_encode)
        except:
            print("Error: unable to fecth data")
            logger.error("Error: unable to fecth data")
        # 关闭数据库连接
        db.close()
        return fname_list, face_encode_list
        print('query from db done')
        logger.info('query from db done')

    def get_max_width_from_bounding_boxes(self, boxes):
        max_width = 0
        for i, bb in enumerate(boxes):
            x = bb[0]
            y = bb[1]
            x1 = bb[2]
            y1 = bb[3]
            if (max_width < (abs(x - x1))):
                max_width = abs(x - x1)
        return max_width

    # 人脸区域框检测
    def detectFaceBoundingBox_mtcnn(self, img):
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold,
                                                          self.factor)  # 利用dect_face检测人脸

        margin= self.margin
        detect_multiple_faces = self.detect_multiple_faces
        # 这里的bounding_boxes实质上是指四个点 四个点连起来构成一个框
        if len(bounding_boxes) < 1:
            print("can't detect face ")  # 当识别不到脸型的时候,不保留
            logger.info("can't detect face ")
            return None
            # bounding_boxes = np.array([[0, 0, img_size[0], img_size[1]]])

        bbox = []
        nrof_faces = bounding_boxes.shape[0]
        img_size = np.asarray(img.shape)[0:2]
        if (detect_multiple_faces):
            det_arr = []
            det = bounding_boxes[:, 0:4]
            for i in range(nrof_faces):
                det_arr.append(np.squeeze(det[i]))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                bbox.append(bb)
        else:
            det = np.squeeze(bounding_boxes[0, 0:4])
            # 这里是为检测到的人脸框加上边界
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            bbox.append(bb)
        return bbox

        # 人脸区域框检测

    def load_and_align_candiate_data(self, image_paths, image_size, margin, pnet, rnet, onet):

        tmp_image_paths = copy.copy(image_paths)
        img_list = []
        for image in tmp_image_paths:
            img = misc.imread(os.path.expanduser(self.path + image), mode='RGB')
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, pnet, rnet, onet, self.threshold,
                                                              self.factor)
            if len(bounding_boxes) < 1:
                image_paths.remove(image)
                print("can't detect face, remove ", image)
                continue
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
        # images = np.stack(img_list)
        return img_list

    def load_and_align_candiate_data_service(self, image_paths, image_size, margin, pnet, rnet, onet):

        tmp_image_paths = copy.copy(image_paths)
        img_list = []
        for image in tmp_image_paths:
            print("load_and_align_candiate_data_service - info,img=", image)
            img = misc.imread(image, mode='RGB')
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, pnet, rnet, onet, self.threshold,
                                                              self.factor)
            if len(bounding_boxes) < 1:
                image_paths.remove(image)
                print("can't detect face, remove ", image)
                continue
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
        # images = np.stack(img_list)
        return img_list
    # 预生成特征向量，保存至数据库，后续直接数据库获取即可
    def save_face_feature_to_db(self):
        print("save_face_feature_to_db start")
        with tf.Graph().as_default():
            with tf.Session() as sess:
                image_files = self.get_candidate_person_pics()

                candiate_img_list = self.load_and_align_candiate_data(image_files, self.image_size, self.margin, self.pnet,
                                                                      self.rnet, self.onet)
                assert len(candiate_img_list) > 0
                img_list = []
                img_list.extend(candiate_img_list)
                images = np.stack(img_list)

                # Run forward pass to calculate embeddings
                feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                emb = sess.run(self.embeddings, feed_dict=feed_dict)

                img_files_set = []
                img_files_set.extend(image_files)
                nrof_images = len(img_files_set)

                print('Images:')
                for i in range(nrof_images):
                    print('%1d: %s' % (i, img_files_set[i]))
                print('')

                # 打开数据库连接
                db = pymysql.connect(self.dburl, self.username, self.password, self.dbname, charset='utf8')
                try:
                    # 使用cursor()方法获取操作游标
                    cursor = db.cursor()
                    # SQL 插入语句
                    for i in range(nrof_images):
                        f_encode = '#'.join(map(str, emb[i, :]))
                        # f_name = os.path.splitext(img_files_set[i])[0]
                        f_name = str(img_files_set[i]).split('/')[0]
                        f_file_name = img_files_set[i]
                        sql = "INSERT INTO face_data(f_name,f_encode,f_file_name)  VALUES('%s','%s','%s')" % (
                            f_name, f_encode, f_file_name)
                        print('sql=', sql)
                        # 执行sql语句
                        cursor.execute(sql)
                        db.commit()
                except:
                    # 如果发生错误则回滚
                    db.rollback()
                # 关闭数据库连接
                db.close()
                print('save_face_feature_to_db save to db done')

        # 预生成特征向量，保存至数据库，后续直接数据库获取即可
    def save_face_feature_to_db_from_input(self, image_files, name):
        print("save_face_feature_to_db from input start, img_files=", image_files, " name=", name)
        with self.sess.graph.as_default():
            with self.sess.as_default():

                candiate_img_list = self.load_and_align_candiate_data_service(image_files, self.image_size, self.margin,
                                                                      self.pnet,
                                                                      self.rnet, self.onet)
                print("save_face_feature_to_db_from_input len(candiate_img_list)=", len(candiate_img_list))
                if(len(candiate_img_list)!=1):
                    return False,"save img must be single face."

                #assert len(candiate_img_list) > 0
                img_list = []
                img_list.extend(candiate_img_list)
                images = np.stack(img_list)

                # Run forward pass to calculate embeddings
                feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                emb = self.sess.run(self.embeddings, feed_dict=feed_dict)

                img_files_set = []
                img_files_set.extend(image_files)
                nrof_images = len(img_files_set)

                print('Images:')
                for i in range(nrof_images):
                    print('%1d: %s' % (i, img_files_set[i]))
                print('')

                # 打开数据库连接
                db = pymysql.connect(self.dburl, self.username, self.password, self.dbname, charset='utf8')
                try:
                    # 使用cursor()方法获取操作游标
                    cursor = db.cursor()
                    # SQL 插入语句
                    for i in range(nrof_images):
                        f_encode = '#'.join(map(str, emb[i, :]))
                        # f_name = os.path.splitext(img_files_set[i])[0]
                        f_name = name#str(img_files_set[i]).split('/')[0]
                        f_file_name = img_files_set[i]
                        sql = "INSERT INTO face_data(f_name,f_encode,f_file_name)  VALUES('%s','%s','%s')" % (
                            f_name, f_encode, f_file_name)
                        print('sql=', sql)
                        # 执行sql语句
                        cursor.execute(sql)
                        db.commit()
                except:
                    # 如果发生错误则回滚
                    db.rollback()
                # 关闭数据库连接
                db.close()
                #重新加载
                self.fname_list, self.face_encode_list = self.query_face_feature_from_db()
                print('save_face_feature_to_db save to db done')
                return True,"success"

    def do_emotion_reginition_process(self, face_image_gray):
        # 加载第一个模型
        with sess1.as_default():
            with g1.as_default():
                emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                resized_img = cv2.resize(face_image_gray, (48, 48), interpolation=cv2.INTER_AREA)
                # cv2.imwrite(str(index)+'.png', resized_img)
                image = resized_img.reshape(1, 1, 48, 48)
                list_of_list = self.emotion_model.predict(image, batch_size=1, verbose=1)
                angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
                res = [angry, fear, happy, sad, surprise, neutral]
                return emotion_labels[res.index(max(res))]

    def do_face_reginition_process_from_input(self, filepath):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                # 处理传入的图片
                frame = misc.imread(filepath)
                res=[]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ori_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.ndim == 2:
                    gray = to_rgb(gray)
                img = gray[:, :, 0:3]
                bboxs = self.detectFaceBoundingBox_mtcnn(img)
                if (bboxs != None):
                    #im = self.fpe.doFaceEstimater(frame)
                    # 人脸处理开始
                    for i, bb in enumerate(bboxs):
                        x = bb[0]
                        y = bb[1]
                        x1 = bb[2]
                        y1 = bb[3]

                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                        reginized_name = None
                        emotion_detect = None
                        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]

                        if (self.emotion_reginition_mode):
                            # print('index ================', index)
                            t = ori_img[bb[1]:bb[3], bb[0]:bb[2]]
                            emotion_detect = self.do_emotion_reginition_process(t)

                        # 图像处理
                        img_list = []
                        aligned = misc.imresize(cropped, (self.image_size, self.image_size),
                                                interp='bilinear')
                        prewhitened = facenet.prewhiten(aligned)
                        img_list.append(prewhitened)
                        images = np.stack(img_list)

                        # Run forward pass to calculate embeddings
                        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
                        # 获取当前检测出的人脸编码值
                        current_emb = emb[0, :]

                        # person_name = knn.predict([current_emb])[0]
                        # print("***********************knn classifier person name result:", person_name)

                        # 识别过程
                        distance_list = []
                        candi_len = len(self.face_encode_list)
                        for id in range(candi_len):
                            dist = np.sqrt(
                                np.sum(np.square(np.subtract(current_emb, self.face_encode_list[id]))))
                            distance_list.append(dist)

                        print("distance list", "=" * 30, distance_list)
                        X = softmax_label([distance_list])
                        max_index = np.argmax(X, axis=1)[0]
                        prob = (str)(round(100 * (X[0][max_index]), 2))
                        min_distance = distance_list[max_index]

                        if (min_distance < self.min_face_distance):
                            reginized_name = os.path.splitext(self.fname_list[max_index])[0]
                            reginized_name = reginized_name + "(" + prob + "%)"

                        else:
                            reginized_name = 'unknown'
                        print('reginized person is:', reginized_name)

                        recognition_title = reginized_name
                        if (emotion_detect is not None):
                            recognition_title = reginized_name + "(" + emotion_detect + ")"
                        obj =DetectObj(i,bb,recognition_title)
                        res.append(obj)
            return res


# if __name__ == "__main__":
#
#     path = os.getcwd() + config.path
#
#     fg = face_reginition(config.dburl, config.username, config.password, config.dbname, config.face_recognition_model,
#                          path, config.face_emotion_model, config.face_estimator_model)
#
#     if (config.init_database):
#         # 初始数据库数据
#         fg.save_face_feature_to_db()
#     else:
#         fg.do_face_reginition_process()
