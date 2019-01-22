# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         InputObject
# Description:  
# Author:       huanghongyi
# Date:         2019/1/21
# -------------------------------------------------------------------------------
import json
import numpy as np

class DetectObj(json.JSONEncoder):
    def __init__(self, id, pos, name=None):
        self.id = id
        seq=[]
        for i in range(4):
            seq.append(str(pos[i]))
        self.pos = "@".join(seq)
        self.name = name

    def setName(self, name):
        self.name = name

    def setPos(self, pos):
        self.pos = pos


class InputEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DetectObj):
            return obj.name
        return json.JSONEncoder.default(self, obj)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
