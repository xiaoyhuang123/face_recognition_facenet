# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         demo_flask
# Description:  
# Author:       huanghongyi
# Date:         2019/1/21
# -------------------------------------------------------------------------------
from flask import Flask
from flask import make_response, Response, render_template, jsonify, request, make_response, send_from_directory, abort
import json
import config
import os
from face_reginition_service import face_reginition
from InputObject import DetectObj, NumpyEncoder
import cv2

app = Flask(__name__)

img_url = "/img_data/"
label_img_url =""
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])

app.config['UPLOAD_FOLDER'] = "img_data"
basedir = os.path.abspath(os.path.dirname(__file__))

# 加载人脸识别服务
path = os.getcwd() + config.path
fg = face_reginition(config.dburl, config.username, config.password, config.dbname, config.face_recognition_model,
                     path, config.face_emotion_model, config.face_estimator_model)


def imgInputParse(para_dict):
    imgs = para_dict["imgs"]
    img_list = json.loads(imgs)
    res = []
    for item in img_list:
        obj = DetectObj(item['id'], json.loads(item['imgArray']))
        res.append(obj)
    return res


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def check_contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


@app.route('/upload')
def upload_test():
    return render_template('index.html')


@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        if os.path.isfile(os.path.join('img_data', filename)):
            return send_from_directory('img_data', filename, as_attachment=True)
        pass


# show photo
@app.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass


@app.route('/up_photo', methods=['post'])
def up_photo():
    img = request.files.get('photo')

    # 判断文件名是否存在中文
    if (not allowed_file(img.filename)):
        content = json.dumps({"error_code": "1001", "msg": "Img format illegal!"})
        return Response_headers(content)
    if (check_contain_chinese(img.filename)):
        content = json.dumps({"error_code": "1002", "msg": "Img name contain chinese charcter!"})
        return Response_headers(content)

    username = str(request.form.get("name"))
    path = basedir + img_url
    file_path = path + img.filename
    if ((username is not None) and (username.strip() != None)):
        user_pic_path = basedir+"/person_pic/"+username
        if(not os.path.exists(user_pic_path)):
            os.mkdir(user_pic_path)
        file_path=user_pic_path+"/"+img.filename
        img.save(file_path)
        print('上传头像成功,name=', username)
        ret,msg = save_face_process(file_path, username)
        response = json.dumps({"result": ret, "msg":msg})
    else:
        img.save(file_path)
        response = face_process(file_path)
        response = json.dumps([d.__dict__ for d in response])
    print("response =====", response)
    resp = Response_headers(response)
    return resp


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    if (request.method == 'POST'):
        # POST:
        # request.form获得所有post参数放在一个类似dict类中,to_dict()是字典化
        # 单个参数可以通过request.form.to_dict().get("xxx","")获得
        datax = request.form.to_dict()
    elif (request.method == 'GET'):
        # GET:
        # request.args获得所有get参数放在一个类似dict类中,to_dict()是字典化
        # 单个参数可以通过request.args.to_dict().get('xxx',"")获得
        datax = request.args.to_dict()

    else:
        content = json.dumps({"error_code": "1001"})
        resp = Response_headers(content)
        return resp

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    d = json.dumps(frame, cls=NumpyEncoder)
    print("frame dumps====", d)

    datax['imgs'] = "[{'id':1,'imgArray':'" + d + "'}]"

    detect_inputs = imgInputParse(datax)
    res = fg.do_face_reginition_process_from_input(detect_inputs)
    resp = Response_headers(res)
    return resp


def face_process(filePath):
    p =filePath # basedir + filePath
    result = fg.do_face_reginition_process_from_input(p)
    return result


def save_face_process(filePath, name):
    p = [] #basedir + filePath
    p.append(filePath)
    result, msg = fg.save_face_feature_to_db_from_input(p, name)
    return result, msg


def Response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.errorhandler(403)
def page_not_found(error):
    content = json.dumps({"error_code": "403"})
    resp = Response_headers(content)
    return resp


@app.errorhandler(404)
def page_not_found(error):
    content = json.dumps({"error_code": "404"})
    resp = Response_headers(content)
    return resp


@app.errorhandler(400)
def page_not_found(error):
    content = json.dumps({"error_code": "400"})
    # resp = Response(content)  
    # resp.headers['Access-Control-Allow-Origin'] = '*'  
    resp = Response_headers(content)
    return resp
    # return "error_code:400"  


@app.errorhandler(410)
def page_not_found(error):
    content = json.dumps({"error_code": "410"})
    resp = Response_headers(content)
    return resp


@app.errorhandler(500)
def page_not_found(error):
    content = json.dumps({"error_code": "500"})
    resp = Response_headers(content)
    return resp


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
