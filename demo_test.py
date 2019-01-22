# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         demo_test
# Description:  
# Author:       huanghongyi
# Date:         2019/1/22
# -------------------------------------------------------------------------------

from InputObject import DetectObj,InputEncoder
import json


if __name__ == "__main__":
    pos=[1,2,3,4]
    obj = DetectObj(1,pos,"hhy")
    res=[]
    res.append(obj)

    s= json.dumps([d.__dict__ for d in res])

    print(s)


