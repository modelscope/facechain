from __future__ import annotations
import os
import sys
sys.path.append("../../")
from functools import partial

import gradio as gr
from dotenv import load_dotenv
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MSPromptGenerator, PromptGenerator
from modelscope_agent.retrieve import ToolRetrieval
from gradio_chatbot import ChatBot
# from mock_llm import MockLLM
from facechain_tools import StyleSearchTool
import copy

from modelscope.utils.config import Config

PROMPT_START = "你好！我是你的FacechainAgent，很高兴为你提供服务。首先，我想了解你对想要创作的写真照有什么大概的想法？"


SYSTEM_PROMPT = """<|system|>: 你现在扮演一个Facechain Agent，不断和用户沟通创作想法，询问用户写真照风格，最后生成搜索到的风格类型返回给用户。当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复。
\n<tool_list>\n"""

INSTRUCTION_TEMPLATE = """【多轮对话历史】

Human: 给我生成一个写真照。

Assistant: 好的，请问你想要什么风格的写真照？

Human: 我想要赛博朋克风。

Assistant: 明白了，我将为你找到需要的风格类型。

<|startofthink|>```JSON\n{\n   "api_name": "style_search",\n    "parameters": {\n      "text": "我想要赛博朋克风。"\n   }\n}\n```<|endofthink|>
我为你找到的风格类型名字是赛博朋克(Cybernetics punk)，该风格文件位置在/content/modelscope-agent/demo/story_agent/style/Cybernetics_punk.json。

【角色扮演要求】
上面多轮角色对话是提供的创作一个写真照风格要和用户沟通的样例，请按照上述的询问步骤来引导用户完成风格的生成，每次只回复对应的内容，不要生成多轮对话。记住只回复用户当前的提问，不要生成多轮对话，回复不要包含<|user|>后面的内容。

"""

KEY_TEMPLATE = """（注意：请参照上述的多轮对话历史流程，但不要生成多轮对话，回复不要包含<|user|>的内容。）"""
#KEY_TEMPLATE = ""

MAX_SCENE = 4

load_dotenv('../../config/.env', override=True)

os.environ['TOOL_CONFIG_FILE'] = '../../config/cfg_tool_template.json'
os.environ['MODEL_CONFIG_FILE'] = '../../config/cfg_model_template.json'
os.environ['OUTPUT_FILE_DIRECTORY'] = './tmp'
os.environ['MODELSCOPE_API_TOKEN'] = 'c70a097b-50bd-42da-9d45-23bed2121eab'
os.environ['DASHSCOPE_API_KEY'] = 'uwjIui5vzfMXRGfWzdU5hkPdE0FJTFFW95425EAEDCCB11ED9809620D7200B5B8'
os.environ['OPENAI_API_KEY'] = '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'

IMAGE_TEMPLATE_PATH = [
    'img_example/1.png',
    'img_example/2.png',
]


with open(
        os.path.join(os.path.dirname(__file__), 'main.css'), "r",
        encoding="utf-8") as f:
    MAIN_CSS_CODE = f.read()

with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:

    max_scene = MAX_SCENE

    with gr.Row():
        gr.HTML(
            """<h1 align="left" style="min-width:200px; margin-top:0;">Facechain Agent</h1>"""
        )
        status_display = gr.HTML(
            "", elem_id="status_display", visible=False, show_label=False)

    with gr.Row(elem_id="container_row").style(equal_height=True):

        with gr.Column(scale=6):
            
            #story_content = gr.Textbox(label='故事情节', lines=4, interactive=False)
            # story_content = ""
            output_image = [None] * max_scene
            output_text = [None] * max_scene

            for i in range(0, max_scene, 2):
                with gr.Row():
                    with gr.Column():
                        output_image[i] = gr.Image(
                            label=f'示例图片{i + 1}',
                            interactive=False,
                            height=200,
                            visible=False,
                            show_progress=False)
                        output_text[i] = gr.Textbox(
                            label=f'故事情节{i + 1}', lines=2, interactive=False, visible=False, show_progress=False)
                    with gr.Column():
                        output_image[i + 1] = gr.Image(
                            label=f'示例图片{i +2}', interactive=False, height=200, visible=False, show_progress=False)
                        output_text[i + 1] = gr.Textbox(
                            label=f'故事情节{i + 2}', lines=2, interactive=False, visible=False, show_progress=False)

        with gr.Column(min_width=470, scale=6, elem_id='settings'):

            chatbot = ChatBot(
                elem_id="chatbot",
                elem_classes=["markdown-body"],
                show_label=False,
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

            # gr.Examples(
            #     examples=['给我生成一个超级向日葵刺猬的故事', '每个段落故事里面都加上超级向日葵刺猬', '可以的，故事生成的不错，我很喜欢！', '卡通画风格'],
            #     inputs=[user_input],
            #     examples_per_page=20,
            #     label="示例",
            #     elem_id="chat-examples")

            # steps = gr.Slider(
            #     minimum=1,
            #     maximum=max_scene,
            #     value=1,
            #     step=1,
            #     label='生成绘本的数目',
            #     interactive=True)
            #steps = 4

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
    style_path="facechain/styles/leosamsMoonfilm_filmGrain20"
    # print_story_tool = PrintStoryTool()
    # show_img_example_tool = ShowExampleTool(IMAGE_TEMPLATE_PATH)
    # image_generation_tool = ImageGenerationTool(output_image, output_text, tool_cfg)
    style_search_tool=StyleSearchTool(style_path)
    additional_tool_list = {
        # print_story_tool.name: print_story_tool,
        # show_img_example_tool.name: show_img_example_tool,
        # image_generation_tool.name: image_generation_tool,
        style_search_tool.name: style_search_tool
    }

    agent = AgentExecutor(
        llm,
        tool_cfg,
        prompt_generator=prompt_generator,
        tool_retrieval=False,
        additional_tool_list=additional_tool_list)

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
        print("user_input: ",user_input)
        print("response: ",response)
        chatbot[-1] = (user_input, response)
        yield chatbot
    
        
        # chatbot[-1] = (user_input, response)
        # yield chatbot
    
    # ---------- 事件 ---------------------

    stream_predict_input = [user_input, chatbot]
    stream_predict_output = [chatbot]

    clean_outputs_start = ['', gr.update(value=[(None, PROMPT_START)])] + [None] * max_scene + [''] * max_scene
    clean_outputs = ['', gr.update(value=[])] + [None] * max_scene + [''] * max_scene
    clean_outputs_target = [user_input, chatbot, *output_image, *output_text]
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
    demo.launch(show_api=False, share=True)
