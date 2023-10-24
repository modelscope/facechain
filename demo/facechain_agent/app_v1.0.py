from __future__ import annotations
import os
import sys
sys.path.append('../../')
from functools import partial
import json
import shutil
import slugify
import glob
from torch import multiprocessing
import gradio as gr
from dotenv import load_dotenv
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MSPromptGenerator, PromptGenerator
from modelscope_agent.retrieve import ToolRetrieval
from gradio_chatbot import ChatBot
from help_tool import StyleSearchTool, FaceChainFineTuneTool, FaceChainInferenceTool
import copy
from facechain.train_text_to_image_lora import prepare_dataset
from modelscope.utils.config import Config
import uuid
import dashscope
from dashscope.audio.tts import SpeechSynthesizer

PROMPT_START = "你好，我是FaceChainAgent，可以帮你生成写真照片。请告诉我你需要的风格的名字。"

SYSTEM_PROMPT = """<|system|>: 你现在扮演一个Facechain Agent，不断和用户沟通创作想法，询问用户写真照风格，最后生成搜索到的风格类型返回给用户。当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复。
\n<tool_list>\n"""

INSTRUCTION_TEMPLATE = """【多轮对话历史】
<|user|>: 给我生成一个赛博朋克风的吧

<|assistant|>: 好的，我将为您找到这个风格类型。
正在搜索风格类型：
<|startofthink|>```JSON\n{\n   "api_name": "style_search_tool",\n    "parameters": {\n      "text": "我想要赛博朋克风。"\n   }\n}\n```<|endofthink|>

<|startofexec|>```JSON\n{"result": {"name": "style_search_tool", "value": "赛博朋克(Cybernetics punk)", file_path: "../../styles/leosamsMoonfilm_filmGrain20/Cybernetics_punk.json"}}\n```<|endofexec|>

我已为你找到的风格类型名字是赛博朋克(Cybernetics punk)。

现在我需要你提供1-3张照片，请点击图片上传按钮上传你的照片。上传完毕后在对话框里告诉我你已经上传好照片了。\n\n</s>

<|user|>: 我的照片上传好了。

<|assistant|>: 收到，我需要几分钟时间训练你上传的照片，然后再生成您的赛博朋克风格写真照。

正在训练人物lora中：
<|startofthink|>```JSON\n{\n   "api_name": "facechain_finetune_tool",\n    "parameters": {}\n}\n```<|endofthink|>

人物lora训练完成。正在生成你选择的赛博朋克风格写真照中：
<|startofthink|>```JSON\n{\n   "api_name": "facechain_inference_tool",\n    "parameters": {\n   "matched_style_file_path": "../../styles/leosamsMoonfilm_filmGrain20/Cybernetics_punk.json"\n  }\n}\n```<|endofthink|>

写真照已经生成完毕！你还可以继续生成这类风格的照片或者更换一个风格\n\n</s>

【角色扮演要求】
上面多轮角色对话是提供的创作一个写真照风格要和用户沟通的样例，请按照上述的询问步骤来引导用户完成风格的生成，每次只回复对应的内容，不要生成多轮对话。记住只回复用户当前的提问，不要生成多轮对话，回复不要包含<|user|>后面的内容。
"""


INSTRUCTION_TEMPLATE1 = """
<|user|>: 我想要换个古风风格。

<|assistant|>: 好的，我将首先搜索相关风格，然后再为您生成古风风格的写真
<|startofthink|>```JSON\n{\n   "api_name": "style_search_tool",\n    "parameters": {\n      "text": "换一个古风的吧"\n   }\n}\n```<|endofthink|>

我为你搜索到的风格是古风风格(Old style)。
我现在将用前面你上传的照片和新选择的风格生成写真照。
生成写真照中：
<|startofthink|>```JSON\n{\n   "api_name": "facechain_inference_tool",\n    "parameters": {\n   "matched_style_file_path": "../../styles/leosamsMoonfilm_filmGrain20/Old_style.json"\n  }\n}\n```<|endofthink|>

古风写真已经生成完毕！你还可以继续生成这类风格的照片或者更换一个风格\n\n</s>

【角色扮演要求】
上面多轮角色对话是提供的创作一个写真照风格要和用户沟通的样例，请按照上述的询问步骤来引导用户完成风格的生成，每次只回复对应的内容，不要生成多轮对话。记住只回复用户当前的提问，不要生成多轮对话，回复不要包含<|user|>后面的内容。

"""


KEY_TEMPLATE = " "
load_dotenv('../config/.env', override=True)
os.environ['TOOL_CONFIG_FILE'] = '../config/cfg_tool_template.json'
os.environ['MODEL_CONFIG_FILE'] = '../config/cfg_model_template.json'
os.environ['OUTPUT_FILE_DIRECTORY'] = './tmp'
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.base_http_api_url = 'https://poc-dashscope.aliyuncs.com/api/v1'
dashscope.base_websocket_api_url = 'https://poc-dashscope.aliyuncs.com/api-ws/v1/inference'

style_paths = ["../../styles/leosamsMoonfilm_filmGrain20", "../../styles/MajicmixRealistic_v6"]
styles = []
for folder_path in style_paths:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            data = json.load(f)
            styles.append(data)

with open(
        os.path.join(os.path.dirname(__file__), 'main.css'), "r",
        encoding="utf-8") as f:
    MAIN_CSS_CODE = f.read()

# ----------agent 对象初始化--------------------

tool_cfg_file = os.getenv('TOOL_CONFIG_FILE')
model_cfg_file = os.getenv('MODEL_CONFIG_FILE')

tool_cfg = Config.from_file(tool_cfg_file)
model_cfg = Config.from_file(model_cfg_file)

# model_name = 'modelscope-agent-7b'
# model_name = 'modelscope-agent'
model_name = 'http_llm'
llm = LLMFactory.build_llm(model_name, model_cfg)

prompt_generator = MSPromptGenerator(
    system_template=SYSTEM_PROMPT,
    instruction_template=INSTRUCTION_TEMPLATE)

prompt_generator1 = MSPromptGenerator(
    system_template=SYSTEM_PROMPT,
    instruction_template=INSTRUCTION_TEMPLATE1)

def add_file(history, files, uuid_str: str):
    if not uuid_str:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid_str = 'facechain_agent'
    history = history + [((file.name,), None) for file in files]
    # task_history = task_history + [((file.name,), None) for file in files]
    file_paths = []
    filtered_list = []
    print(history)
    file_paths = [item[0][0] for item in history if item[0] != None]
    if len(file_paths) == 0:
        raise ValueError("照片上传失败")
    print("#####", file_paths)
    filtered_list = [item for item in file_paths if '.jpg' in item or '.png' in item]
    print("#####", filtered_list)
    uuid = 'qw'
    # shutil.rmtree(f"./{uuid}", ignore_errors=True)
    base_model_path = 'cv_portrait_model'
    revision = 'v2.0'
    sub_path = "film/film"
    uuid_str=uuid_str[:8]
    print("#####切割后uuid_str",uuid_str)
    output_model_name = uuid_str
    output_model_name = slugify.slugify(output_model_name)
    instance_data_dir = os.path.join('./', base_model_path, output_model_name)
    shutil.rmtree(instance_data_dir, ignore_errors=True)
    try:
        prepare_dataset(filtered_list, instance_data_dir)
    except Exception as e:
        raise e("预处理图片数据出错")
    return history


def reset_user_input():
    return gr.update(value="")


def text_to_speech(text):
    result = SpeechSynthesizer.call(model='sambert-zhichu-v1',
                                    text=text,
                                    sample_rate=48000,
                                    format='wav')
    return result.get_audio_data()

    if result.get_audio_data() is not None:
        with open('tts.wav', 'wb') as f:
            f.write(result.get_audio_data())


def init(uuid_str, state):
    if not uuid_str:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid_str = 'facechain_agent'
    #print('##############################', uuid_str)
    uuid_str=uuid_str[:8]
    #print("#####切割后uuid_str",uuid_str)
    style_search_tool = StyleSearchTool(style_paths)
    facechain_finetune_tool = FaceChainFineTuneTool(uuid_str)  # 初始化lora_name,区分不同用户
    facechain_inference_tool = FaceChainInferenceTool(uuid_str)
    additional_tool_list = {
        style_search_tool.name: style_search_tool,
        facechain_finetune_tool.name: facechain_finetune_tool,
        facechain_inference_tool.name: facechain_inference_tool
    }
    agent = AgentExecutor(
        llm,
        tool_cfg,
        prompt_generator=prompt_generator,
        tool_retrieval=False,
        additional_tool_list=additional_tool_list,
        # knowledge_retrieval=knowledge_retrieval
    )
    agent.set_available_tools(additional_tool_list.keys())
    state['agent'] = agent
    state['additional_tool_list'] = additional_tool_list


with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:
    uuid_str = gr.Textbox(label="modelscope_uuid", visible=False)
    state = gr.State({})
    demo.load(init, inputs=[uuid_str, state], outputs=[])
    # 生成随机的 UUID（和uuid=‘qw'不一样）

    with gr.Row():
        gr.Markdown(
            "# <center> \N{fire} FaceChain Potrait Generation ([Github star it here](https://github.com/modelscope/facechain/tree/main) \N{whale},   [Paper cite it here](https://arxiv.org/abs/2308.14256) \N{whale})</center>")
    with gr.Row():   
        gr.Markdown(
            "##### <center> 本项目仅供学习交流，请勿将模型及其制作内容用于非法活动或违反他人隐私的场景。(This project is intended solely for the purpose of technological discussion, and should not be used for illegal activities and violating privacy of individuals.)</center>")
    with gr.Row():
        gr.Markdown(
            """ <img src=https://gw.alicdn.com/imgextra/i1/O1CN01y1wvKm1HdHow9Gx1Y_!!6000000000780-0-tps-1800-654.jpg>""")
    with gr.Row():
        with gr.Column():
            gr.Markdown(""" 🌈 🌈 🌈

                        ## 你好，我是FaceChain Agent，可以帮你生成写真照片。

                        ## 下图是各类风格的展示图，你可以在这先挑选你喜欢的风格。

                        ## 然后在下方的聊天框里与我交流吧，一起来生成美妙的写真照！

                        """)
            with gr.Row():
                gallery = gr.Gallery(value=[(os.path.join("../../", item["img"]), item["name"]) for item in styles],
                                     elem_id='gallery', show_label=False
                                     ).style(object_fit='contain', preview=True, columns=6)
    with gr.Row(elem_id="container_row").style(equal_height=True):
        with gr.Column(scale=8, elem_classes=["chatInterface", "chatDialog", "chatContent"]):
            with gr.Row(elem_id='chat-container'):
                chatbot = ChatBot(value=[[None, PROMPT_START]],
                                  elem_id="chatbot",
                                  elem_classes=["markdown-body"],
                                  show_label=False,
                                  )

            with gr.Row(elem_id="chat-bottom-container"):
                with gr.Column(min_width=70, scale=1):
                    clear_session_button = gr.Button(
                        "清除", elem_id='clear_session_button', default_value=True)
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="一起来自由生成写真照吧～",
                        elem_id="chat-input").style(container=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("发送", variant="primary")
                with gr.Column(min_width=110, scale=1):
                    upload_button = gr.UploadButton("上传照片", file_types=["image"], file_count="multiple")
                with gr.Column(min_width=110, scale=1):
                    regenerate_button = gr.Button(
                        "重新生成", elem_id='regenerate_button')
            gr.Examples(
                examples=['我想要牛仔风', '我想要凤冠霞帔风', '我的照片上传好了', '我现在想换个风格，我想要工作风'],
                inputs=[user_input],
                label="示例",
                elem_id="chat-examples")


    def facechain_agent(*inputs):

        user_input = inputs[0]
        chatbot = inputs[1]
        state = inputs[2]
        agent = state['agent']

        chatbot.append((user_input, None))
        yield chatbot
        history_image = []

        def update_component(exec_result, history):
            exec_result = exec_result['result']
            name = exec_result.pop('name')
            if name == 'facechain_inference_tool':
                single_path = exec_result['single_path']
                #print("########_______single_path", single_path)

                image_files = glob.glob(os.path.join(single_path, '*.jpg'))
                image_files += glob.glob(os.path.join(single_path, '*.png'))
                #print("########_______image_files", image_files)

                history = [(None, (file,)) for file in image_files]
                # task_history  = task_history + [(None,(file,)) for file in image_files]
            else:
                history = []
                # task_history  = task_history
            return history

        response = ''
        for frame in agent.stream_run(user_input + KEY_TEMPLATE, remote=True):
            is_final = frame.get("frame_is_final")
            llm_result = frame.get("llm_text", "")
            exec_result = frame.get('exec_result', '')
            # print(frame)
            history = []
            llm_result = llm_result.split("<|user|>")[0].strip()
            if len(exec_result) != 0:
                history = update_component(exec_result, chatbot)
                print("#########________history", history)
                frame_text = " "
            else:
                # action_exec_result
                frame_text = llm_result
                response = f'{response}\n{frame_text}'
                chatbot[-1] = (user_input, response)
            if history != []:
                history_image = history

            yield chatbot
        try:
            if history_image != []:
                try:
                    agent = AgentExecutor(
                        llm,
                        tool_cfg,
                        prompt_generator=prompt_generator1,
                        tool_retrieval=False,
                        additional_tool_list=state["additional_tool_list"],
                        # knowledge_retrieval=knowledge_retrieval
                    )
                    agent.set_available_tools(state["additional_tool_list"].keys())
                    inputs[2]['agent'] = agent
                except Exception as e:
                    import traceback
                    print(f'error {e} with detail {traceback.format_exc()}')

                for item in history_image:
                    chatbot.append(item)
                    yield chatbot

        except:
            pass


    # ---------- 事件 ---------------------

    stream_predict_input = [user_input, chatbot, state]
    stream_predict_output = [chatbot]

    clean_outputs_start = ['', gr.update(value=[(None, PROMPT_START)])]
    clean_outputs = ['', gr.update(value=[])]
    clean_outputs_target = [user_input, chatbot]
    user_input.submit(
        facechain_agent,
        inputs=stream_predict_input,
        outputs=stream_predict_output,
        show_progress=True)
    user_input.submit(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)
    submitBtn.click(
        facechain_agent,
        stream_predict_input,
        stream_predict_output,
        show_progress=True
    )
    submitBtn.click(reset_user_input, [], [user_input])
    regenerate_button.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)
    regenerate_button.click(
        facechain_agent,
        stream_predict_input,
        stream_predict_output,
        show_progress=True)


    def clear_session(state):
        agent = state['agent']
        agent.reset()


    clear_session_button.click(fn=clear_session, inputs=[state], outputs=[])
    clear_session_button.click(
        fn=lambda: clean_outputs_start, inputs=[], outputs=clean_outputs_target)
    upload_button.upload(add_file, inputs=[chatbot, upload_button, uuid_str], outputs=[chatbot], show_progress=True)
    # chatbot[-1] = ((None, PROMPT_START))
demo.title = "Facechian Agent 🎁"
if __name__ == "__main__":
    # print(multiprocessing.get_start_method())
    multiprocessing.set_start_method('spawn')
    demo.queue(status_update_rate=1).launch(share=True)
