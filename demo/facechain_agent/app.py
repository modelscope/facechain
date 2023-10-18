from __future__ import annotations
import os
#安装后可以注释或者删除
os.system('pip install modelscope_agent-0.1.0-py3-none-any.whl')#根据modelscope agent 项目生成.whl包
os.system('pip install gradio==3.29.0')
import sys
sys.path.append("../../")
from functools import partial
import json
import shutil
import slugify
import glob
from torch import multiprocessing
import PIL.Image
import gradio as gr
from dotenv import load_dotenv
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MSPromptGenerator, PromptGenerator
from modelscope_agent.retrieve import ToolRetrieval
from gradio_chatbot import ChatBot
#from mock_llm import MockLLM
from help_tool import StyleSearchTool,FaceChainFineTuneTool,FaceChainInferenceTool
import copy
from facechain.train_text_to_image_lora import prepare_dataset
from modelscope.utils.config import Config
import uuid


# 生成随机的 UUID（和uuid=‘qw'不一样）
random_uuid = uuid.uuid4()

# 将 UUID 转换为字符串
uuid_str = str(random_uuid)
image_num = 3
PROMPT_START = "你好！我是你的FacechainAgent，很高兴为你提供服务。首先，我想了解你对想要创作的写真照有什么大概的想法？"

SYSTEM_PROMPT = """<|system|>: 你现在扮演一个Facechain Agent，不断和用户沟通创作想法，询问用户写真照风格，最后生成搜索到的风格类型返回给用户。当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复。
\n<tool_list>\n"""

INSTRUCTION_TEMPLATE = """【多轮对话历史】

<|user|>: 给我生成一个写真照。

<|assistant|>: 好的，请问您想要什么风格的写真照？

<|user|>: 我想要赛博朋克风。

<|assistant|>: 好的，我将为您找到这个风格类型。
正在搜索风格类型：
<|startofthink|>```JSON\n{\n   "api_name": "style_search_tool",\n    "parameters": {\n      "text": "我想要赛博朋克风。"\n   }\n}\n```<|endofthink|>

现在我需要你提供1-3张照片，请点击图片上传按钮上传你的照片。上传完毕后在对话框里告诉我你已经上传好照片了。

<|user|>: 我的照片上传好了。

<|assistant|>: 收到，我需要几分钟时间训练你上传的照片，然后再生成您的赛博朋克风格写真照。

正在训练人物lora中：
<|startofthink|>```JSON\n{\n   "api_name": "facechain_finetune_tool",\n    "parameters": {}\n}\n```<|endofthink|>

人物lora训练完成。正在生成你选择的赛博朋克风格写真照中：
<|startofthink|>```JSON\n{\n   "api_name": "facechain_inference_tool",\n    "parameters": {\n   "matched_style_file_path": "../../styles/leosamsMoonfilm_filmGrain20/Cybernetics_punk.json"\n  }\n}\n```<|endofthink|>

写真照已经生成完毕，如果喜欢赶紧下载打印吧！你可以继续生成这类风格的照片或者跟我说换一个风格

<|user|>: 再换一个古风风格吧

<|assistant|>: 好的，我将首先搜索相关风格，然后再为您生成古风风格的写真

正在搜索风格类型：
<|startofthink|>```JSON\n{\n   "api_name": "style_search_tool",\n    "parameters": {\n      "text": "再换一个古风风格吧"\n   }\n}\n```<|endofthink|>

正在生成古风风格写真照中：
<|startofthink|>```JSON\n{\n   "api_name": "facechain_inference_tool",\n    "parameters": {\n   "matched_style_file_path": "../../styles/leosamsMoonfilm_filmGrain20/Old_style.json"\n  }\n}\n```<|endofthink|>

古风写真已经生成完毕，赶紧下载打印吧！你还可以继续生成这类风格的照片或者跟我说换一个风格
【角色扮演要求】
上面多轮角色对话是提供的创作一个写真照风格要和用户沟通的样例，请按照上述的询问步骤来引导用户完成风格的生成，每次只回复对应的内容，不要生成多轮对话。记住只回复用户当前的提问，不要生成多轮对话，回复不要包含<|user|>后面的内容。

"""

# INSTRUCTION_TEMPLATE = """【多轮对话历史】

# <|user|>: 给我生成一个写真照。

# <|assistant|>: 好的，请问你想要什么风格的写真照？

# <|user|>: 我想要赛博朋克风。

# <|assistant|>: 好的，我将为你找到这个风格类型。
# 正在搜索风格类型：

# <|startofthink|>JSON\n{\n   "api_name": "style_search_tool",\n    "parameters": {\n      "text": "我想要赛博朋克风。"\n   }\n}\n<|endofthink|>

# <|startofexec|>```JSON\n{"result": {"name": "style_search_tool", "value": "赛博朋克(Cybernetics punk)", file_path: "../../styles/leosamsMoonfilm_filmGrain20/Cybernetics_punk.json"}}\n```<|endofexec|>

# 我为你找到的风格类型名字是赛博朋克(Cybernetics punk)。现在我需要你提供1-3张照片，请点击上传照片按钮上传你的照片。
# 上传完毕后在对话框里告诉我你已经上传好照片了。

# <|user|>: 我的照片上传好了。

# <|assistant|>: 收到，我需要几分钟时间训练你上传的照片。
# 正在训练人物lora中：
# <|startofthink|>```JSON\n{\n   "api_name": "facechain_finetune_tool",\n    "parameters": {}\n}\n```<|endofthink|>

# 人物lora训练完成。你要使用你之前选择的赛博朋克(Cybernetics punk)风格生成写真照吗，还是你要更换风格？

# <|user|>: 不换，就用这个风格。

# <|assistant|>: 好的，我将为你生成赛博朋克(Cybernetics punk)风格的写真照。这将需要几分钟时间，请耐心等待。
# 正在生成写真照中：
# <|startofthink|>```JSON\n{\n   "api_name": "facechain_inference_tool",\n    "parameters": {\n   "matched_style_file_path": "/home/wsco/wyj2/facechain-agent/styles/leosamsMoonfilm_filmGrain20/Cybernetics_punk.json"\n  }\n}\n```<|endofthink|>

# 写真照已经生成完毕，如果喜欢赶紧下载打印吧！

# <|user|>: 我现在想换个风格，我想要工作风。

# <|assistant|>:好的，我将更新你想要的风格类型。
# 正在搜索风格类型：<|startofthink|>```JSON\n{\n   "api_name": "style_search_tool",\n    "parameters": {\n      "text": "我现在想换个风格，我想要工作风。"\n   }\n}\n```<|endofthink|>

# 我为你找到的风格类型名字是工作服(Working suit)。
# 我现在将用前面你上传的照片和新选择的风格生成写真照。
# 正在生成写真照中：
# <|startofthink|>```JSON\n{\n   "api_name": "facechain_inference_tool",\n    "parameters": {\n   "matched_style_file_path": "/home/wsco/wyj2/facechain-agent/styles/leosamsMoonfilm_filmGrain20/Working_suit.json"\n  }\n}\n```<|endofthink|>

# 写真照已经生成完毕，如果喜欢赶紧下载打印吧！
# 【角色扮演要求】
# 上面多轮角色对话是提供的创作一个写真照风格要和用户沟通的样例，请按照上述的询问步骤来引导用户完成风格的生成，每次只回复对应的内容，不要生成多轮对话。记住只回复用户当前的提问，不要生成多轮对话，回复不要包含<|user|>后面的内容。

# """

KEY_TEMPLATE = """（注意：请参照上述的多轮对话历史流程，但不要生成多轮对话，回复不要包含<|user|>的内容。）"""


load_dotenv('../config/.env', override=True)
os.environ['TOOL_CONFIG_FILE'] = '../config/cfg_tool_template.json'
os.environ['MODEL_CONFIG_FILE'] = '../config/cfg_model_template.json'
os.environ['OUTPUT_FILE_DIRECTORY'] = './tmp'
#write your key here or write in ../config/.env
os.environ['MODELSCOPE_API_TOKEN'] = 'xxx'
os.environ['DASHSCOPE_API_KEY'] = 'xxx'
os.environ['OPENAI_API_KEY'] = 'xxx'

style_paths=["../../styles/leosamsMoonfilm_filmGrain20","../../styles/MajicmixRealistic_v6"]
styles=[]
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

def add_file(history,files,task_history,output_model_name):
        history = history+[((file.name,), None) for file in files]
        task_history = task_history + [((file.name,), None) for file in files]
        file_paths = []
        filtered_list = []
        print(history)
        file_paths =[item[0][0] for item in history]
        print("#####",file_paths)
        filtered_list = [item for item in file_paths if '.jpg' in item or '.png' in item]
        print("#####",filtered_list)
        uuid = 'qw'
        #shutil.rmtree(f"./{uuid}", ignore_errors=True)
        base_model_path = 'ly261666/cv_portrait_model'
        revision = 'v2.0'
        sub_path = "film/film"
        output_model_name = uuid_str
        output_model_name = slugify.slugify(output_model_name)
    
        instance_data_dir = os.path.join('./', uuid, 'training_data', base_model_path, output_model_name)
        shutil.rmtree(instance_data_dir, ignore_errors=True)   
        prepare_dataset(filtered_list, instance_data_dir)
        return history,task_history
def reset_user_input():
        return gr.update(value="")

with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:
   
    with gr.Row():
        gr.HTML(
            """<h1 align="left" style="min-width:200px; margin-top:0;">Facechain Agent</h1>"""
        )
        status_display = gr.HTML(
            "", elem_id="status_display", visible=False, show_label=False)


    with gr.Row():
        gr.Markdown(""" 🌈 🌈 🌈
                    
                    ## 你好，我是FaceChain Agent，可以帮你生成写真照片。
                    
                    ## 右图是各类风格的展示图，你可以在这先挑选你喜欢的风格。
                    
                    ## 然后在下方的聊天框里与我交流吧，一起来生成美妙的写真照！
                    
                    """)
        gallery = gr.Gallery(value=[(os.path.join("../../",item["img"]), item["name"]) for item in styles],
                                        elem_id='gallery',
                                        ).style(object_fit='contain',preview=True,columns=6)
    with gr.Row(elem_id="container_row").style(equal_height=True):
        with gr.Column(scale=8, elem_classes=["chatInterface", "chatDialog", "chatContent"]):
            with gr.Row(elem_id='chat-container'):
                chatbot = ChatBot(
                    elem_id="chatbot",
                    elem_classes=["markdown-body"],
                    show_label=True,
                    )
                task_history = gr.State([])
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
                    upload_button = gr.UploadButton("上传照片", file_types=["image"],file_count="multiple")
                with gr.Column(min_width=110, scale=1):
                    regenerate_button = gr.Button(
                        "重新生成", elem_id='regenerate_button')
                    
            gr.Examples(
            examples=['我想要写真照','我想要凤冠霞帔风','我的照片上传好了','不换，就用这个风格','我现在想换个风格，我想要工作风'],
            inputs=[user_input],
            label="示例",
            elem_id="chat-examples")


    # ----------agent 对象初始化--------------------

    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE')
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE')

    tool_cfg = Config.from_file(tool_cfg_file)
    model_cfg = Config.from_file(model_cfg_file)

    model_name = 'openai'
    #model_name = 'modelscope-agent-7b'
    llm = LLMFactory.build_llm(model_name, model_cfg)
    #llm = MockLLM()

    prompt_generator = MSPromptGenerator(
        system_template=SYSTEM_PROMPT,
        instruction_template=INSTRUCTION_TEMPLATE)


    # tools 

    style_search_tool=StyleSearchTool(style_paths)
    facechain_finetune_tool=FaceChainFineTuneTool(uuid_str)#初始化lora_name,区分不同用户
    facechain_inference_tool=FaceChainInferenceTool(uuid_str)
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
        #knowledge_retrieval=knowledge_retrieval
        )

    agent.set_available_tools(additional_tool_list.keys())

    def story_agent(*inputs):

        global agent
        user_input = inputs[0] 
        chatbot = inputs[1]
        task_history = inputs[2]
        output_component = list(inputs[3:])
        def reset_component():
            for i in range(image_num):
                output_component[i+1] = gr.Image.update(visible=False)
        
        chatbot.append((user_input, None))
        task_history.append((user_input, None))
        #chatbotd(user_input)
        yield chatbot,*output_component,task_history
        
        def update_component(exec_result,history,task_history):
            exec_result = exec_result['result']
            name = exec_result.pop('name')
            if name == 'facechain_inference_tool':
                #print("#############-------------")
                single_path = exec_result['single_path']
                image_files = glob.glob(os.path.join(single_path, '*.jpg'))
                image_files += glob.glob(os.path.join(single_path, '*.png'))
                # output_component[0] = gr.Image.update(image_files[0])
                # output_component[1] = gr.Image.update(image_files[1])
                # output_component[2] = gr.Image.update(image_files[2])
                history = [(None,(file,)) for file in image_files] 
                task_history  = task_history + [(None,(file,)) for file in image_files] 
            else:
                history = [] 
                task_history  = task_history
            return history,task_history       
        response = ''        
        for frame in agent.stream_run(user_input+KEY_TEMPLATE, remote=True):
            is_final = frame.get("frame_is_final")
            llm_result = frame.get("llm_text", "")
            exec_result = frame.get('exec_result', '') 
            #print(frame)
            history = []
            llm_result = llm_result.split("<|user|>")[0].strip()
            if len(exec_result) != 0:
                [history,task_history]=update_component(exec_result,chatbot,task_history)
                frame_text = " "
                # response = f'{response}\n{frame_text}'
                # chatbot[-1] = (user_input, response)
                # task_history[-1] = (user_input, response)
            else:
                # action_exec_result
                frame_text = llm_result
                response = f'{response}\n{frame_text}'
                chatbot[-1] = (user_input, response)
                task_history[-1] = (user_input, response)
            if history != []:
                 history_image = history
            task_history = task_history[-10:]
            yield chatbot,*copy.deepcopy(output_component),task_history
        try:
            if history_image != []:
                print()
                for item in history_image:
                    chatbot.append(item)
                    yield chatbot,*copy.deepcopy(output_component),task_history           
        except:
            pass

    
        
   
    # ---------- 事件 ---------------------

    stream_predict_input = [user_input, chatbot,task_history]
    stream_predict_output = [chatbot,task_history]

    clean_outputs_start = ['', gr.update(value=[(None, PROMPT_START)])]+[None] * image_num + [''] * image_num
    clean_outputs = ['', gr.update(value=[])]+[None] * image_num + [''] * image_num
    clean_outputs_target = [user_input, chatbot]
    user_input.submit(
        story_agent,
        inputs=stream_predict_input,
        outputs=stream_predict_output,
        show_progress=True)
    user_input.submit(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)
    submitBtn.click(
        story_agent,
        stream_predict_input,
        stream_predict_output,
        show_progress=True
    )
    submitBtn.click(reset_user_input, [], [user_input])
    regenerate_button.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)
    regenerate_button.click(
        story_agent,
        stream_predict_input,
        stream_predict_output,
        show_progress=True)

    def clear_session():
        agent.reset()

    clear_session_button.click(fn=clear_session, inputs=[], outputs=[])
    clear_session_button.click(
        fn=lambda: clean_outputs_start, inputs=[], outputs=clean_outputs_target)
    upload_button.upload(add_file, inputs=[chatbot,upload_button,task_history], outputs=[chatbot,task_history],show_progress=True) 
    # chatbot.append((None, PROMPT_START))
demo.title = "Facechian Agent 🎁"
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    demo.queue(status_update_rate=1).launch()
