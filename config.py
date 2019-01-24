# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         config
# Description:  
# Author:       huanghongyi
# Date:         2019/1/9
# -------------------------------------------------------------------------------

# mysql数据库
dburl = '127.0.0.1'
username = 'root'
password = '123456'
dbname = 'face_db'

# mysql数据库
sqllite_dburl = 'db/face_db.db'
sqllite_dbname = 'face_db'

# 候选集合图像集合
path = "/person_pic/"

# 预训练模型名
face_recognition_model = "model/20170512-110547"
face_emotion_model = "model/model_emo/model.h5"
face_estimator_model="/model/shape_predictor_68_face_landmarks.dat"
init_database = False

# 掩码   从左至右：第一位：人脸识别模式  第二位：情感分析模式
# 0：关闭    1：开启
runing_mode="11"
