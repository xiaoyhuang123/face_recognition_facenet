# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         db_sqllite
# Description:  
# Author:       huanghongyi
# Date:         2019/1/24
# -------------------------------------------------------------------------------

import sqlite3
db_name="face_db.db"

class DB_sqllite:
    def __init__(self, dbname=None):
        self.db_name =db_name
        if(dbname is not None):
            self.db_name = dbname

    def do_excute(self, sql):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        #c.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) VALUES (1, 'Paul', 32, 'California', 20000.00 )");
        c.execute(sql)
        conn.commit()
        print("Records created successfully")
        conn.close()

    def select_face_data(self,sql):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        #sql = "SELECT f_name,f_encode,f_file_name FROM face_data"
        #c.execute(sql)
        cursor = c.execute(sql)

        res=[]
        for row in cursor:
            f_encode = row[1]
            f_name =row[0]
            f_file_name = row[2]
            s={}
            s['f_name']=f_name
            #s['f_encode'] = f_encode
            #s['f_file_name'] = f_file_name
            res.append(s)
        print("Records created successfully")
        conn.close()
        return res


if __name__ =="__main__":
#    dbs = DB_sqllite()
#     sql = '''CREATE TABLE `face_data` (
#   id INTEGER PRIMARY KEY AUTOINCREMENT,
#   f_name varchar(100) DEFAULT NULL,
#   f_encode varchar(5000) DEFAULT NULL,
#   f_file_name varchar(500) DEFAULT NULL
# ) '''
#     dbs.do_excute(sql)

    # sql = "delete FROM face_data"
    # dbs.do_excute(sql)

    # f_encode = '############'
    # f_name = 'nnnaammee'
    # f_file_name = 'filename'
    # sql = "INSERT INTO face_data(f_name,f_encode,f_file_name)  VALUES('%s','%s','%s')" % (
    #     f_name, f_encode, f_file_name)
    # sql ="INSERT INTO face_data(f_name,f_encode,f_file_name)  VALUES('hhhh','0.12871999#0.042515647#-0.07001011#-0.04559021#0.018119883#-0.0626277#0.010017009#-0.1006516#-0.076684944#-0.14394587#0.098607816#-0.045504455#-0.09181649#-0.116873845#-0.029120551#0.052283544#0.07550767#-0.10162355#0.01841207#0.0016249279#-0.06517963#0.12030605#-0.040557332#-0.031909365#-0.04488555#0.04210974#-0.018570423#0.019130893#0.04393351#0.11569854#0.0692108#0.113738135#-0.02019653#-0.005327142#0.064692944#-0.011463832#0.02052937#0.15552722#0.23707078#0.109164484#-0.14020896#0.043368053#-0.0607957#-0.07570844#-0.12960342#-0.02414273#-0.070821166#0.028748732#-0.0626469#0.15125547#0.060380794#0.051232252#0.11153051#-0.0024863905#0.064187154#0.16469924#-0.036270868#0.007973281#-0.042854793#-0.058323182#-0.12269997#-0.19943094#-0.013776127#0.02875282#0.028029244#0.0804599#0.05821741#0.02519684#-0.117643125#0.059567887#0.05502333#-0.06373049#0.01947802#-0.032679062#-0.05742306#0.09795217#-0.09411536#0.069694646#-0.20504911#-0.10218936#0.02558201#0.013836766#-0.047950417#0.2289699#0.1121688#-0.08337461#0.028430613#-0.16477768#-0.1727845#-0.06455362#0.10498094#-0.06856867#-0.06342411#-0.15127686#-0.051740076#0.04592567#-0.019487785#-0.07423245#-0.093798764#-0.23578832#-0.1642845#-0.05196007#0.031029362#-0.049992036#-0.040225103#-0.029919669#0.06769695#0.008815748#0.06259948#-0.09023787#0.08971792#-0.028440982#0.10320522#0.068655476#0.11690336#0.0028995546#-0.012025621#0.04522703#-0.0013433176#0.05690525#0.07730082#-0.11254093#0.12507212#-0.044691905#0.0016616797#-0.06891928#-0.14629957#-0.025639486','E:\jdcloud_ai\code learning\hhy_hithub\face_recognition_facenet/person_pic/hhhh/5ac3d9c0-a3e7-4096-8190-842fa290ec2b.jpg')"
    # dbs.do_excute(sql)
    #
    #
    # sql = "SELECT f_name,f_encode,f_file_name FROM face_data"
    #
    # res = dbs.select_face_data(sql)
    # print("res =", res)
    dbs = DB_sqllite()
    sql = "SELECT f_name,f_encode,f_file_name FROM face_data"
    sql = "Delete  FROM face_data"
    dbs.do_excute(sql)
    print(dbs.select_face_data(sql))

    try:
        conn = sqlite3.connect(db_name)
        # 使用cursor()方法获取操作游标
        c = conn.cursor()
        # 执行SQL语句
        cursor = c.execute(sql)
        # 获取所有记录列表
        #print("fetch data size=", len(cursor))
        for row in cursor:
            fname = row[0]
            face_encode_str = row[1]
            face_encode = face_encode_str.split('#')
            face_encode = list(map(float, face_encode))

            face_file_path = row[2]
    except:
        print("Error: unable to fecth data")
    # 关闭数据库连接
    conn.close()

