from flask import Flask, request
import sys
sys.path.append('../../')
from demo.facechain_agent.biz import add_file, get_and_run_agent
import uuid
import os
app = Flask(__name__)


# ----------agent 对象初始化--------------------


@app.route('/')
def index():
    return ''


# @app.route('/custom_response', methods=['POST'])
# def custom_response():
#     file = request.files['image']
#
#     message = request.data.decode('utf-8')
#     message = json.loads(message)
#     user_id = message['user_id']
#     inputs = message['inputs']
#     uuid_str = user_id
#
#     user_input = inputs[0]
#
#     inputs = list(inputs)  # 将字符串转换成字符数组
#     length = len(inputs)  # 获取数组的长度
#
#     if length == 1:
#         chatbot = []
#         task_history = []
#     else:
#         chatbot = inputs[1]
#         task_history = inputs[2]
#     # 如果用户没有选择文件，浏览器会发送一个空文件
#     if file.filename == '':
#         return 'No selected file'
#
#     if file:
#         # 保存文件到本地
#         path = os.path.join('./source', user_id, str(uuid.uuid4()))
#         file.save('D:\\PycharmProjects\\facechain-agent-gradio\\111.png')
#         add_file( user_id, path)
#
#     response = get_and_run_agent(user_input, user_id, chatbot)
#
#     chatbot.append((user_input, None))
#     task_history.append((user_input, None))
#
#     return response
#

@app.route('/facechain_agent', methods=['POST'])
def upload_pic():
    # message = request.form.decode('utf-8')

    if request.form is None:
        return "message is none"

    # message = json.loads(message)
    user_id = request.form['user_id']
    if user_id is None or user_id == '':
        return "user_id is none"

    user_input = request.form['inputs']
    # inputs = request.form['inputs']

    if request.files:
        file = request.files['image']
        if file:
            if file.filename == '':
                return 'No selected file'
            # 保存文件到本地
            path = os.path.join('.\\source', user_id)
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, str(uuid.uuid4()) + '.png')
            file.save(path)
            add_file(user_id, path)

    response = get_and_run_agent(user_input, user_id, "")

    return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=6006)
