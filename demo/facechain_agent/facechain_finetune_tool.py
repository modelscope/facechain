import os
import shutil
import gradio as gr
from modelscope.utils.config import Config
from modelscope_agent.agent import AgentExecutor
from modelscope_agent.llm import LLMFactory
from modelscope_agent.prompt import MSPromptGenerator

import app

import slugify

from modelscope_agent.tools import Tool

from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn

from torch import multiprocessing

class FaceChainFineTuneTool(Tool):
    description = "模型微调工具，根据用户提供的图片训练出Lora"
    name = "facechain_finetune_tool"
    parameters: list = [{
        'name': 'text',
        'description': '用户输入的文本信息',
        'required': True
    }]

    def __init__(self):
        super().__init__()
        self.base_model_path = 'ly261666/cv_portrait_model'
        self.revision = 'v2.0'
        self.sub_path = "film/film"
        # 这里固定了Lora的名字,重新训练会覆盖原来的
        self.lora_name = "person1"

    def _remote_call(self, *args, **kwargs):
        pass

    def _local_call(self, *args, **kwargs):
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        instance_images = kwargs['instance_images']

        # train lora
        _train_lora(uuid, self.lora_name, instance_images, self.base_model_path, self.revision, self.sub_path)

        result = {'lora_name': self.lora_name, 'uuid': uuid, 'msg': "训练完成"}
        return {'result': result}


def _train_lora(uuid, output_model_name, instance_images, base_model_path, revision, sub_path):
    output_model_name = slugify.slugify(output_model_name)

    # mv user upload data to target dir
    instance_data_dir = os.path.join('./', uuid, 'training_data', base_model_path, output_model_name)

    if not os.path.exists(f"./{uuid}"):
        os.makedirs(f"./{uuid}")
    work_dir = f"./{uuid}/{base_model_path}/{output_model_name}"

    # if os.path.exists(work_dir):
    #     raise gr.Error("人物lora名称已存在。(This character lora name already exists.)")

    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(instance_data_dir, ignore_errors=True)
    prepare_dataset([img for img in instance_images], output_dataset_dir=instance_data_dir)
    data_process_fn(instance_data_dir, True)

    app.train_lora_fn(base_model_path=base_model_path,
                      revision=revision,
                      sub_path=sub_path,
                      output_img_dir=instance_data_dir,
                      work_dir=work_dir,
                      photo_num=len(instance_data_dir))

    return base_model_path, revision, sub_path, instance_data_dir, work_dir


SYSTEM_PROMPT = """<|system|>: 你现在扮演一个Facechain Agent，帮助用户画图，先询问用户绘图风格，然后要求用户上传原始图片，再根据用户所传图片生成用户lora。当前对话可以使用的插件信息如下，请自行判断是否需要调用tool来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。然后你需要根据插件API调用结果生成合理的答复。
\n<tool_list>\n"""
KEY_TEMPLATE = """（注意：请参照上述的多轮对话历史流程，但不要生成多轮对话，回复不要包含<|user|>的内容。）"""
INSTRUCTION_TEMPLATE = """
[对话历史]
Human: 请给我训练一个lora

Assistant: 我需要你提供1-3张照片，用于训练你的数字分身，然后再生成对应的数字写真。请点击图片上传按钮。

Human: 已经上传

Assistant: 收到，我需要10分钟训练并生成，你可以过10分钟再回来界面。正在训练中：<|startofthink|>```JSON\n{\n   "api_name": "facechain_finetune_tool",\n    "parameters": {\n "uuid": "0", "instance_images": "pic_001"\n   }\n}\n```<|endofthink|>
"""

os.environ['TOOL_CONFIG_FILE'] = '../../config/cfg_tool_template.json'
os.environ['MODEL_CONFIG_FILE'] = '../../config/cfg_model_template.json'
os.environ['OPENAI_API_KEY'] = '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'

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    tool_cfg_file = os.getenv('TOOL_CONFIG_FILE')
    model_cfg_file = os.getenv('MODEL_CONFIG_FILE')

    tool_cfg = Config.from_file(tool_cfg_file)
    model_cfg = Config.from_file(model_cfg_file)

    model_name = 'openai'
    llm = LLMFactory.build_llm(model_name, model_cfg)

    tool = FaceChainFineTuneTool()
    tool._local_call(args=None, instance_images=["/mnt/workspace/v.jpg"])

    tool_list = {
        tool.name: tool,
    }

    prompt_generator = MSPromptGenerator(
        system_template=SYSTEM_PROMPT,
        instruction_template=INSTRUCTION_TEMPLATE)

    agent = AgentExecutor(llm, additional_tool_list=tool_list, prompt_generator=prompt_generator, tool_retrieval=False)

    while True:
        user_input = input("")

        if user_input.lower() == "exit":
            print("对话结束。")
            break

        # 在这里添加你的代码来处理用户输入并生成回应
        response = user_input

        r = True
        for frame in agent.stream_run(KEY_TEMPLATE + user_input, remote=r):
            is_final = frame.get("frame_is_final")
            llm_result = frame.get("llm_text", "")
            exec_result = frame.get('exec_result', '')
            llm_result = llm_result.split("<|user|>")[0].strip()
            if len(exec_result) != 0:
                # llm_result
                frame_text = ' '
            else:
                # action_exec_result
                frame_text = llm_result
            response = f'{response}\n{frame_text}'
            print(frame)
