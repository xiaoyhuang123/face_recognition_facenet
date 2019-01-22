# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         data_parse_test
# Description:  
# Author:       huanghongyi
# Date:         2019/1/21
# -------------------------------------------------------------------------------
import pickle
import numpy as np
import json
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# a= np.arange(12).reshape(3,4)
# print(a)
# s=json.dumps(a,cls=NumpyEncoder)
# print(s)
#
# datax = "[{'id':1,'imgArray':" + s + "}]"
#
# aa = json.loads(datax)
# print(aa)


basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)