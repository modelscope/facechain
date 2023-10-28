import base64

from flask import Flask, request, jsonify
import sys

sys.path.append('../../')
from demo.facechain_agent.biz import add_file, run_facechain_agent
import uuid
import os
from torch import multiprocessing

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return '123'


@app.route('/facechain_agent', methods=['POST'])
def facechain_agent():
    req_from = request.form
    req_file = request.files
    user_id = req_from['user_id']
    user_input = req_from['inputs']

    # 参数校验
    if req_from is None:
        return "message is none"
    if user_id is None or user_id == '':
        return "user_id is none"

    #  保存request中的图片
    if req_file:
        file = req_file['image']
        if file:
            if file.filename == '':
                return 'No selected file'
            save_req_pic(file, user_id)

    # 核心逻辑
    response, image_paths = run_facechain_agent(user_input, user_id)

    # 如果返回值包含图片
    # image_paths = ["/root/000.jpg", "/root/26fb0b2c-425e-465d-a4cb-ddbd4ac8d85c.png"]
    if image_paths:
        images = []  #
        for image_path in image_paths:
            # 打开图片文件
            with open(image_path, "rb") as image_file:
                image_file = image_file.read()

                encoded_image = base64.b64encode(image_file)
                encoded_image_str = encoded_image.decode("utf-8")
                images.append(encoded_image_str)

        return jsonify({'images': images})

    return response


def save_req_pic(file, user_id):

    path = os.path.join('./source_file', user_id)
    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, str(uuid.uuid4()) + '.png')
    file.save(path)
    add_file(user_id, path)


# todo 1.定时删除 用户agent
# todo 2.优化图片保存逻辑
# todo 3.共用 prompt
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    app.run(host='0.0.0.0', debug=True, port=6006)
