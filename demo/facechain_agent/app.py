from __future__ import annotations
import os
import sys
sys.path.append("../../")
#sys.path.append("/home/wsco/wyj2/modelscope-agent-1")
from functools import partial
import json
import shutil
import slugify
import PIL.Image
import gradio as gr
from dotenv import load_dotenv
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MSPromptGenerator, PromptGenerator
from modelscope_agent.retrieve import ToolRetrieval
from gradio_chatbot import ChatBot
#from mock_llm import MockLLM
from help_tool import StyleSearchTool,FaceChainFineTuneTool
import copy
from facechain.train_text_to_image_lora import prepare_dataset,data_process_fn,get_rot
from modelscope.utils.config import Config

PROMPT_START = "你好！我是你的FacechainAgent，很高兴为你提供服务。首先，我想了解你对想要创作的写真照有什么大概的想法？"


SYSTEM_PROMPT = """<|system|>: 你现在扮演一个Facechain Agent，不断和用户沟通创作想法，询问用户写真照风格，最后生成搜索到的风格类型返回给用户。当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复。
\n<tool_list>\n"""

INSTRUCTION_TEMPLATE = """【多轮对话历史】

Human: 给我生成一个写真照。

Assistant: 好的，请问你想要什么风格的写真照？

Human: 我想要赛博朋克风。

Assistant: 明白了，我将为你找到需要的风格类型。

<|startofthink|>```JSON\n{\n   "api_name": "style_search_tool",\n    "parameters": {\n      "text": "我想要赛博朋克风。"\n   }\n}\n```<|endofthink|>
我为你找到的风格类型名字是赛博朋克(Cybernetics punk)。

现在我需要你提供1-3张照片，请点击图片上传按钮上传你的照片。上传完毕后在对话框里告诉我你已经上传好照片了。

Human: 我的照片上传好了。

Assistant: 收到，我需要10分钟训练并生成，你可以过10分钟再回来界面。

正在训练人物lora中：<|startofthink|>```JSON\n{\n   "api_name": "facechain_finetune_tool",\n    "parameters": {\n \n   }\n}\n```<|endofthink|>
人物lora训练完成。是否根据你刚才选的赛博朋克风格生成写真照？还是你要更换风格吗？


【角色扮演要求】
上面多轮角色对话是提供的创作一个写真照风格要和用户沟通的样例，请按照上述的询问步骤来引导用户完成风格的生成，每次只回复对应的内容，不要生成多轮对话。记住只回复用户当前的提问，不要生成多轮对话，回复不要包含<|user|>后面的内容。

"""

KEY_TEMPLATE = """（注意：请参照上述的多轮对话历史流程，但不要生成多轮对话，回复不要包含<|user|>的内容。）"""
#KEY_TEMPLATE = ""



load_dotenv('../../config/.env', override=True)

os.environ['TOOL_CONFIG_FILE'] = '../config/cfg_tool_template.json'
os.environ['MODEL_CONFIG_FILE'] = '../config/cfg_model_template.json'
os.environ['OUTPUT_FILE_DIRECTORY'] = './tmp'
os.environ['MODELSCOPE_API_TOKEN'] = 'c70a097b-50bd-42da-9d45-23bed2121eab'
os.environ['DASHSCOPE_API_KEY'] = 'uwjIui5vzfMXRGfWzdU5hkPdE0FJTFFW95425EAEDCCB11ED9809620D7200B5B8'
os.environ['OPENAI_API_KEY'] = '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'

style_path="/home/wsco/wyj2/facechain/styles/leosamsMoonfilm_filmGrain20"
styles=[]
for filename in os.listdir(style_path):
    file_path = os.path.join(style_path, filename)
    with open(file_path,"r") as f:
        data=json.load(f)
        styles.append(data)


with open(
        os.path.join(os.path.dirname(__file__), 'main.css'), "r",
        encoding="utf-8") as f:
    MAIN_CSS_CODE = f.read()
def upload_file(files,current_files):
    
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    #prepare_dataset([img['name'] for img in instance_images], instance_data_dir=instance_data_dir)
    for i, temp_path in enumerate(file_paths):
        image = PIL.Image.open(temp_path)
        image = image.convert('RGB')
        image = get_rot(image)
        # image = image.resize((new_w, new_h))
        # image = image.resize((new_w, new_h), PIL.Image.ANTIALIAS)
        uuid = 'qw'
        shutil.rmtree(f"./{uuid}", ignore_errors=True)
        base_model_path = 'ly261666/cv_portrait_model'
        revision = 'v2.0'
        sub_path = "film/film"
        output_model_name='person1'
        output_model_name = slugify.slugify(output_model_name)
        # mv user upload data to target dir
        instance_data_dir = os.path.join('./', uuid, 'training_data', base_model_path, output_model_name)
        shutil.rmtree(instance_data_dir, ignore_errors=True)
        if not os.path.exists(instance_data_dir):
            os.makedirs(instance_data_dir)
        out_path = f'{instance_data_dir}/{i:03d}.jpg'
        image.save(out_path, format='JPEG', quality=100)
    data_process_fn(instance_data_dir,True)

    print(file_paths)
        
    return file_paths


with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:
    uuid = gr.Text(label="modelscope_uuid", visible=False)
    with gr.Row():
        gr.HTML(
            """<h1 align="left" style="min-width:200px; margin-top:0;">Facechain Agent</h1>"""
        )
        status_display = gr.HTML(
            "", elem_id="status_display", visible=False, show_label=False)

    with gr.Row(elem_id="container_row").style(equal_height=True):
        
        with gr.Column(min_width=470, scale=6, elem_id='settings'):
            gr.Markdown(""" 🌈 你好，我是FaceChain Agent，可以帮你生成写真照片。
                        
                        以下是各类风格的展示图，请挑选你喜欢的风格并在下方的聊天框里与我交流吧。""")
            gallery = gr.Gallery(value=[(os.path.join("/home/wsco/wyj2/facechain",item["img"]), item["name"]) for item in styles],
                                            label="风格(Style)",
                                            allow_preview=False,
                                            columns=5,
                                            elem_id="gallery",
                                            show_share_button=False,
                                            object_fit="contain"
                                            )
            chatbot = ChatBot(
                elem_id="chatbot",
                elem_classes=["markdown-body"],
                show_label=True,
                height=600)
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
                    regenerate_button = gr.Button(
                        "重新生成", elem_id='regenerate_button')
                gr.Examples(
                examples=['我想要写真照','我想要凤冠霞帔风','我的照片上传好了'],
                inputs=[user_input],
                label="示例",
                elem_id="chat-examples")
            with gr.Row():
                instance_images = gr.Gallery()
                with gr.Row(min_width=110, scale=1):
                    upload_button = gr.UploadButton("📁上传图片", file_types=["image"],file_count="multiple")
                    clear_button = gr.Button("清空图片(Clear photos)")
            clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)
            upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images,
                                        queue=False)
            
            #trainer = Trainer()
            # upload_button.click(fn=trainer.run,
            #                     inputs=[instance_images
            #                         ],
            #                     outputs=[output_message])


    # ----------agent 对象初始化--------------------

    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE')
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE')

    tool_cfg = Config.from_file(tool_cfg_file)
    model_cfg = Config.from_file(model_cfg_file)

    model_name = 'openai'
    llm = LLMFactory.build_llm(model_name, model_cfg)
    #llm = MockLLM()

    prompt_generator = MSPromptGenerator(
        system_template=SYSTEM_PROMPT,
        instruction_template=INSTRUCTION_TEMPLATE)

    # model_cfg = {
    #     'modelscope-agent-qwen-7b': {
    #         'model_id': 'damo/MSAgent-Qwen-7B',
    #         'model_revision': 'v1.0.2',
    #         'use_raw_generation_config': True,
    #         'custom_chat': True
    #     }
    # }


    # tools 
    
    model_id = 'damo/nlp_corom_sentence-embedding_chinese-base'
    filepath="/home/wsco/wyj2/modelscope-agent-1/demo/story_agent/style.txt"

    style_search_tool=StyleSearchTool(style_path)
    facechain_finetune_tool=FaceChainFineTuneTool()
    additional_tool_list = {
        style_search_tool.name: style_search_tool,
        facechain_finetune_tool.name:facechain_finetune_tool
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
        chatbot.append((user_input, None))
        #chatbotd(user_input)
        yield chatbot
        response = ''
        
        for frame in agent.stream_run(user_input+KEY_TEMPLATE, remote=True):
            is_final = frame.get("frame_is_final")
            llm_result = frame.get("llm_text", "")
            exec_result = frame.get('exec_result', '') 
            #print(frame)
            llm_result = llm_result.split("<|user|>")[0].strip()
            if len(exec_result) != 0:
                
                frame_text = " "
            else:
                # action_exec_result
                frame_text = llm_result
            response = f'{response}\n{frame_text}'
            # chatbot[-1] = (user_input, response)
            # yield chatbot
        print("user_input: ",user_input)
        print("response: ",response)
        chatbot[-1] = (user_input, response)
        yield chatbot
    
        
        # chatbot[-1] = (user_input, response)
        # yield chatbot
    
    # ---------- 事件 ---------------------

    stream_predict_input = [user_input, chatbot]
    stream_predict_output = [chatbot]

    clean_outputs_start = ['', gr.update(value=[(None, PROMPT_START)])]
    clean_outputs = ['', gr.update(value=[])] 
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
        show_progress=True)
    submitBtn.click(
        fn=lambda: clean_outputs, inputs=[], outputs=clean_outputs_target)

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
  
    # chatbot.append((None, PROMPT_START))
demo.title = "Facechian Agent 🎁"
demo.queue(concurrency_count=10, status_update_rate='auto', api_open=False)
demo.launch(show_api=False, share=False)
