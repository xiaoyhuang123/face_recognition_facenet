# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         utils
# Description:  
# Author:       huanghongyi
# Date:         2019/1/10
# -------------------------------------------------------------------------------
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def knn_clf_for_face_reginition(features, labels, face_feature):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, labels)
    name = knn.predict([face_feature])
    return name[0]


def softmax_label(X):
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    X *= -1
    # X = np.exp(X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

class Logger:
    def __init__(self, path, Flevel=logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # # 设置CMD日志
        # sh = logging.StreamHandler()
        # sh.setFormatter(fmt)
        # sh.setLevel(clevel)
        # 设置文件日志
        fh = logging.FileHandler("log.txt")
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        #self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def war(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)