# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         ResponseVo
# Description:  
# Author:       huanghongyi
# Date:         2019/1/24
# -------------------------------------------------------------------------------
from flask import Response
import json

class ResponseVo:
    def __init__(self, code, msg, data):
        self.code = code
        self.msg = msg
        self.data = data

    def setCode(self, code):
        self.code = code

    def setMsg(self, msg):
        self.msg = msg

    def setData(self, data):
        self.data = data

    def toDict(self):
        res = {}
        res['code'] = self.code
        res['msg'] = self.msg
        res['data'] = self.data
        return res

    def Response_headers(self):
        d = self.toDict()
        content = json.dumps(d)
        print("==Response_headers content==", content)
        resp = Response(content)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
