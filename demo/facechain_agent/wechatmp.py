import base64

from flask import Flask, request, jsonify
import sys

sys.path.append('../../')
from demo.facechain_agent.biz import save_req_pic, run_facechain_agent

from torch import multiprocessing

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'root'


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

    response, image_paths = run_facechain_agent(user_input, user_id)

    # 返回图片
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


# todo 1.定时删除 用户agent
# todo 2.优化图片保存逻辑，基于被注释的 biz.save_req_picV2
# todo 3.共用 prompt
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    app.run(host='0.0.0.0', debug=True, port=6006)
