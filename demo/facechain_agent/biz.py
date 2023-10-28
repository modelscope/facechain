import glob
import os
import uuid
import sys
sys.path.append('../../')
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MSPromptGenerator
from help_tool import StyleSearchTool, FaceChainFineTuneTool, FaceChainInferenceTool
from modelscope.utils.config import Config
import shutil
import slugify
from facechain.train_text_to_image_lora import prepare_dataset, get_rot
from dotenv import load_dotenv
import dashscope
from PIL import Image

style_paths = ["../../styles/leosamsMoonfilm_filmGrain20", "../../styles/MajicmixRealistic_v6"]
random_uuid = uuid.uuid4()
uuid_str = str(random_uuid)

PROMPT_START = "你好，我是FaceChainAgent，可以帮你生成写真照片。请告诉我你需要的风格的名字。"

SYSTEM_PROMPT = """<|system|>: 你现在扮演一个Facechain Agent，不断和用户沟通创作想法，询问用户写真照风格，最后生成搜索到的风格类型返回给用户。当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复。
\n<tool_list>\n"""

KEY_TEMPLATE = " "

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


my_map = {}
def get_agent(user_id):
    agent =my_map.get(user_id)
    if agent is  None:

        tool_cfg_file = os.getenv('TOOL_CONFIG_FILE')
        model_cfg_file = os.getenv('MODEL_CONFIG_FILE')

        tool_cfg = Config.from_file(tool_cfg_file)
        model_cfg = Config.from_file(model_cfg_file)

        # model_name = 'openai'
        # # model_name = 'modelscope-agent-7b'
        # llm = LLMFactory.build_llm(model_name, model_cfg)
        # # llm = MockLLM()
        model_name = 'http_llm'
        llm = LLMFactory.build_llm(model_name, model_cfg)

        prompt_generator = MSPromptGenerator(
            system_template=SYSTEM_PROMPT,
            instruction_template=INSTRUCTION_TEMPLATE)

        # tools

        style_search_tool = StyleSearchTool(style_paths)
        facechain_finetune_tool = FaceChainFineTuneTool(user_id)  # 初始化lora_name,区分不同用户
        facechain_inference_tool = FaceChainInferenceTool(user_id)
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
        my_map[user_id] = agent

    return agent


def add_file(uuid_str, path):

    filtered_list = [path]
    print("#####", filtered_list)

    base_model_path = 'cv_portrait_model'

    output_model_name = uuid_str
    output_model_name = slugify.slugify(output_model_name)

    instance_data_dir = os.path.join('./', base_model_path, output_model_name)

    shutil.rmtree(instance_data_dir, ignore_errors=True)
    prepare_dataset(filtered_list, instance_data_dir)


def get_pic_path(exec_result):
    exec_result = exec_result['result']
    name = exec_result.pop('name')
    if name == 'facechain_inference_tool':
        single_path = exec_result['single_path']

        image_files = glob.glob(os.path.join(single_path, '*.jpg'))
        image_files += glob.glob(os.path.join(single_path, '*.png'))

        return image_files
    return None

def run_facechain_agent(user_input,user_id):
    agent = get_agent(user_id)
    response = ''
    for frame in agent.stream_run(user_input + KEY_TEMPLATE, remote=True):
        is_final = frame.get("frame_is_final")
        llm_result = frame.get("llm_text", "")
        exec_result = frame.get('exec_result', '')
        llm_result = llm_result.split("<|user|>")[0].strip()
        if len(exec_result) != 0:
            image_files = get_pic_path(exec_result)
            if image_files:
                return image_files,"images"
            frame_text = " "
            response = f'{frame_text}'

        else:
            frame_text = llm_result
            if llm_result !="":
                response = f'{response} \n {frame_text}'


    return  response,""


# def save_req_pic(file, user_id):
#
#     # path = os.path.join('./source_file', user_id)
#     # if not os.path.exists(path):
#     #     os.makedirs(path)
#
#     # path = os.path.join(path, str(uuid.uuid4()) + '.png')
#     # file.save(path)
#     # add_file(user_id, path)
#
#     base_model_path = 'cv_portrait_model'
#     output_model_name = user_id
#     output_model_name = slugify.slugify(output_model_name)
#     instance_data_dir = os.path.join('./', base_model_path, output_model_name)
#     shutil.rmtree(instance_data_dir, ignore_errors=True)
#
#     if not os.path.exists(instance_data_dir):
#         os.makedirs(instance_data_dir)
#
#         i=0
#         '''
#         w, h = image.size
#         max_size = max(w, h)
#         ratio =  1024 / max_size
#         new_w = round(w * ratio)
#         new_h = round(h * ratio)
#         '''
#         image = Image.open(file)
#
#         image = image.convert('RGB')
#
#         image = get_rot(image)
#
#         out_path = f'{instance_data_dir}/{i:03d}.jpg'
#         image.save(out_path, format='JPEG', quality=100)
