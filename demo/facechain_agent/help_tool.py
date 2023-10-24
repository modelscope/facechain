import os
import sys
from modelscope_agent.tools import Tool, TextToImageTool
from langchain.embeddings import ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document
import gradio as gr
from difflib import SequenceMatcher
from typing import List
from pathlib import Path
import json
import platform
import random
import shutil
import slugify
import subprocess
import time
import torch
import cv2
import numpy as np

from facechain.inference import data_process_fn
from facechain.utils import snapshot_download
from concurrent.futures import ProcessPoolExecutor
from facechain.inference import GenPortrait
  
class StyleSearchTool(Tool):
    description = ''
    name = 'style_search_tool'
    parameters: list = [{
        'name': 'text',
        'description': '用户输入的想要的风格文本',
        'required': True
    },   
    ]
    def __init__(self, style_path: List[str]):
        self.style_path = style_path
        super().__init__()
    def _remote_call(self, text):
      
      best_match = None  # 用于存储最佳匹配风格
      best_similarity = 0  # 用于存储最佳匹配的相似性度量值
      best_file_path = None  # 用于存储最佳匹配的文件路径
      for style_folder in self.style_path:
    
        # 列出风格文件夹下的所有文件
        files = os.listdir(style_folder)
        files.sort()
        for file in files:
            file_path = os.path.join(style_folder, file)
            with open(file_path, "r") as f:
                data = json.load(f)
                style_name = data.get('name', '')  # 获取风格文件中的名称字段

                # 计算文本与风格名称之间的相似性
                similarity = SequenceMatcher(None, text, style_name).ratio()

                # 如果相似性高于当前最佳匹配，更新最佳匹配
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = style_name
                    best_file_path = file_path  # 更新最佳匹配的文件路径

      result = {
          'name': self.name,
          'value': best_match ,
          'file_path':best_file_path # 返回最相似的风格名称    
      }

      return {'result': result}
    def _remote_call(self, text):
        best_match = None  # 用于存储最佳匹配风格
        best_similarity = 0  # 用于存储最佳匹配的相似性度量值
        best_file_path = None  # 用于存储最佳匹配的文件路径

        for style_folder in self.style_path:
            # 列出风格文件夹下的所有文件
            files = os.listdir(style_folder)
            files.sort()
            for file in files:
                file_path = os.path.join(style_folder, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    style_name = data.get('name', '')  # 获取风格文件中的名称字段

                    # 计算文本与风格名称之间的相似性
                    similarity = SequenceMatcher(None, text, style_name).ratio()

                    # 如果相似性高于当前最佳匹配，更新最佳匹配
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = style_name
                        best_file_path = file_path  # 更新最佳匹配的文件路径

        result = {
            'name': self.name,
            'value': best_match ,
            'file_path':best_file_path # 返回最相似的风格名称  
           }

        return {'result': result}

class FaceChainFineTuneTool(Tool):
    description = "模型微调工具，根据用户提供的图片训练出Lora"
    name = "facechain_finetune_tool"
    parameters: list = []

    def __init__(self,lora_name:str):
        super().__init__()
        self.base_model_path = 'ly261666/cv_portrait_model'
        self.revision = 'v2.0'
        self.sub_path = "film/film"
        # 这里固定了Lora的名字,重新训练会覆盖原来的
        self.lora_name = lora_name
        

    def _remote_call(self):
        uuid = 'qw'
        # train lora
        _train_lora(uuid, self.lora_name, self.base_model_path, self.revision, self.sub_path)

        result = {'name':self.name,'lora_name': self.lora_name, 'uuid': uuid, 'msg': "训练完成"}
        return {'result': result}

    def _local_call(self):
        uuid = 'qw'
        
        # train lora
        _train_lora(uuid, self.lora_name,self.base_model_path, self.revision, self.sub_path)

        result = {'name':self.name,'lora_name': self.lora_name, 'uuid': uuid, 'msg': "训练完成"}
        return {'result': result}
    
def train_lora_fn(base_model_path=None, revision=None, sub_path=None, output_img_dir=None, work_dir=None, photo_num=0):
    torch.cuda.empty_cache()

    lora_r = 4
    lora_alpha = 32
    #max_train_steps = min(photo_num * 200, 800)

    if platform.system() == 'Windows':
        command = [
            'accelerate', 'launch', '../../facechain/train_text_to_image_lora.py',
            f'--pretrained_model_name_or_path={base_model_path}',
            f'--revision={revision}',
            f'--sub_path={sub_path}',
            f'--output_dataset_name={output_img_dir}',
            '--caption_column=text',
            '--resolution=512',
            '--random_flip',
            '--train_batch_size=1',
            '--num_train_epochs=200',
            '--checkpointing_steps=5000',
            '--learning_rate=1.5e-04',
            '--lr_scheduler=cosine',
            '--lr_warmup_steps=0',
            '--seed=42',
            f'--output_dir={work_dir}',
            f'--lora_r={lora_r}',
            f'--lora_alpha={lora_alpha}',
            '--lora_text_encoder_r=32',
            '--lora_text_encoder_alpha=32',
            '--resume_from_checkpoint="fromfacecommon"'
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
            raise RuntimeError("训练失败 (Training failed)")
    else:
        res = os.system(
            f'PYTHONPATH=. accelerate launch ../../facechain/train_text_to_image_lora.py '
            f'--pretrained_model_name_or_path={base_model_path} '
            f'--revision={revision} '
            f'--sub_path={sub_path} '
            f'--output_dataset_name={output_img_dir} '
            f'--caption_column="text" '
            f'--resolution=512 '
            f'--random_flip '
            f'--train_batch_size=1 '
            f'--num_train_epochs=200 '
            f'--checkpointing_steps=5000 '
            f'--learning_rate=1.5e-04 '
            f'--lr_scheduler="cosine" '
            f'--lr_warmup_steps=0 '
            f'--seed=42 '
            f'--output_dir={work_dir} '
            f'--lora_r={lora_r} '
            f'--lora_alpha={lora_alpha} '
            f'--lora_text_encoder_r=32 '
            f'--lora_text_encoder_alpha=32 '
            f'--resume_from_checkpoint="fromfacecommon"')
        if res != 0:
            raise RuntimeError("训练失败 (Training failed)")


def _train_lora(uuid, output_model_name, base_model_path, revision, sub_path):
    output_model_name = slugify.slugify(output_model_name)

    # mv user upload data to target dir
    instance_data_dir = os.path.join('./', base_model_path.split('/')[-1], output_model_name)
    print('################################instance_data_dir', instance_data_dir)
    if not os.path.exists(f"./{uuid}"):
        os.makedirs(f"./{uuid}")
    work_dir = f"./{uuid}/{base_model_path}/{output_model_name}"
    print('################################work_dir', work_dir)

    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        data_process_fn(instance_data_dir, True)
    except Exception as e:
        raise e("提取图片label错误") from e

    train_lora_fn(
        base_model_path=base_model_path,
        revision=revision,
        sub_path=sub_path,
        output_img_dir=instance_data_dir,
        work_dir=work_dir,
        photo_num=len(instance_data_dir)
    )

    return base_model_path, revision, sub_path, instance_data_dir, work_dir

#--------------------------------------
training_done_count = 0
inference_done_count = 0
base_models = [
    {'name': 'leosamsMoonfilm_filmGrain20',
     'model_id': 'ly261666/cv_portrait_model',
     'revision': 'v2.0',
     'sub_path': "film/film"},
    {'name': 'MajicmixRealistic_v6',
     'model_id': 'YorickHe/majicmixRealistic_v6',
     'revision': 'v1.0.0',
     'sub_path': "realistic"},
]
neg_prompt = '(nsfw:2), paintings, sketches, (worst quality:2), (low quality:2), ' \
             'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, bad hand, tattoo, (username, watermark, signature, time signature, timestamp, artist name, copyright name, copyright),'\
             'low res, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, strange fingers, bad hand, mole, ((extra legs)), ((extra hands))'
pos_prompt_with_cloth = 'raw photo, masterpiece, chinese, {}, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, photorealistic, best quality'
pos_prompt_with_style = '{}, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, photorealistic, best quality'

def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image
def launch_pipeline(uuid,
           pos_prompt,
           matched,
           num_images,
           neg_prompt=None,
           base_model_index=0,
           user_model=None,
           lora_choice=None,
           multiplier_style=0.35,
           multiplier_human=0.95,
           pose_model=None,
           pose_image=None,
           ):
    uuid = 'qw'
    character_model='ly261666/cv_portrait_model'#
    # Check character LoRA
    folder_path = f"./{uuid}/{character_model}"
    folder_list = []
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)
    if len(folder_list) == 0:
        raise '没有人物LoRA，请先训练(There is no character LoRA, please train first)!'
    # Check output model
    if user_model is None:
        raise '请选择人物LoRA(Please select the character LoRA)！'
    base_model = base_models[base_model_index]['model_id']
    revision = base_models[base_model_index]['revision']
    sub_path = base_models[base_model_index]['sub_path']
    before_queue_size = 0
    before_done_count = inference_done_count
    style_model = matched['name']
    if matched['model_id'] is None:
        style_model_path = None
    else:
        model_dir = snapshot_download(matched['model_id'], revision=matched['revision'])
        style_model_path = os.path.join(model_dir, matched['bin_file'])
    
    if pose_image is None or pose_model == 0:
        pose_model_path = None
        use_depth_control = False
        pose_image = None
    else:
        model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
        pose_model_path = os.path.join(model_dir, 'model_controlnet/control_v11p_sd15_openpose')
        if pose_model == 1:
            use_depth_control = True
        else:
            use_depth_control = False

    print("-------user_model(也就是人物lora name): ", user_model)

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False
#user_model就是人物lora的name
    instance_data_dir = os.path.join('./', character_model.split('/')[-1], user_model)
    lora_model_path = f'./{uuid}/{character_model}/{user_model}'
    print('################################instance_data_dir', instance_data_dir)
    print('################################lora_model_path', lora_model_path)

    gen_portrait = GenPortrait(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path,
                               multiplier_style, multiplier_human, use_main_model,
                               use_face_swap, use_post_process,
                               use_stylization)

    num_images = min(6, num_images)

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(gen_portrait, instance_data_dir,
                                 num_images, base_model, lora_model_path, sub_path, revision)
        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = inference_done_count
                to_wait = before_queue_size - (cur_done_count - before_done_count)
                print("排队等待资源中, 前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5),
                      None)
            else:
                print("生成中, 请耐心等待(Generating)...", None)
            time.sleep(1)

    outputs = future.result()
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))

    save_dir = os.path.join('./', base_model, user_model)
    if lora_choice == 'preset':
        save_dir = os.path.join(save_dir, 'style_' + style_model[:2])
    else:
        save_dir = os.path.join(save_dir, 'lora_' + os.path.basename(lora_choice).split('.')[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # use single to save outputs
    if not os.path.exists(os.path.join(save_dir, 'single')):
        os.makedirs(os.path.join(save_dir, 'single'))
    single_path=os.path.join(save_dir, 'single')
    for img in outputs:
        # count the number of images in the folder
        num = len(os.listdir(os.path.join(save_dir, 'single')))
        cv2.imwrite(os.path.join(save_dir, 'single', str(num) + '.png'), img)
    
    if len(outputs) > 0:
        result = concatenate_images(outputs)
        if not os.path.exists(os.path.join(save_dir, 'concat')):
            os.makedirs(os.path.join(save_dir, 'concat'))
        num = len(os.listdir(os.path.join(save_dir, 'concat')))
        image_path = os.path.join(save_dir, 'concat', str(num) + '.png')
        cv2.imwrite(image_path, result)#整体图像

        return ("生成完毕(Generation done)!", outputs_RGB,single_path)
    else:
        return ("生成失败, 请重试(Generation failed, please retry)!", outputs_RGB,single_path)

def generate_pos_prompt(matched_style_file, prompt_cloth):
    if matched_style_file is not None:
        # matched = list(filter(lambda style: style_model == style['name'], styles))
        # if len(matched) == 0:
        #     raise ValueError(f'styles not found: {style_model}')
        
        if matched_style_file['model_id'] is None:
            pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
        else:
            pos_prompt = pos_prompt_with_style.format(matched_style_file['add_prompt_style'])
    else:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    return pos_prompt
class FaceChainInferenceTool(Tool):
    description = "根据用户人脸lora和风格lora生成写真"
    name = 'facechain_inference_tool'
    parameters: list = [{
        'name': 'matched_style_file_path',
        'description': '风格文件的位置',
        'required': True
    }]
#user_model 也就是 lora_name
    def __init__(self,user_model:str):
        self.user_model = user_model
        super().__init__()

    def _remote_call(self, matched_style_file_path:str):
        with open (matched_style_file_path,'r') as f:
            matched_style_file = json.load(f)
        pos_prompt = generate_pos_prompt(matched_style_file, matched_style_file['add_prompt_style'])
        print(os.path.dirname(matched_style_file_path))
        if  "leosamsMoonfilm_filmGrain20" in matched_style_file_path:
            base_model_index = 0
        elif  "MajicmixRealistic_v6" in matched_style_file_path:
             base_model_index = 1
        (infer_progress,output_images,single_path)=launch_pipeline(uuid='qw',matched=matched_style_file,pos_prompt=pos_prompt,
               neg_prompt=neg_prompt, base_model_index=base_model_index,
               user_model=self.user_model, num_images=3,
                multiplier_style=0.35,
               multiplier_human=0.95, pose_model=None,
               pose_image=None, lora_choice='preset'
               )
        result = {'name':self.name, 'infer_progress':infer_progress, 'output_images':output_images, 'single_path':single_path}
        return {'result': result}

    def _local_call(self, matched_style_file_path:str):
        with open (matched_style_file_path,'r') as f:
            matched_style_file = json.load(f)
        pos_prompt = generate_pos_prompt(matched_style_file, matched_style_file['add_prompt_style'])
        if  "leosamsMoonfilm_filmGrain20" in matched_style_file_path:
            base_model_index = 0
        elif  "MajicmixRealistic_v6" in matched_style_file_path:
             base_model_index = 1
        (infer_progress,output_images,single_path)=launch_pipeline(uuid='qw',matched=matched_style_file,pos_prompt=pos_prompt,
               neg_prompt=neg_prompt, base_model_index=base_model_index,
               user_model=self.user_model, num_images=3,
                multiplier_style=0.35,
               multiplier_human=0.95, pose_model=None,
               pose_image=None, lora_choice='preset'
               )
        result = {'name':self.name, 
                  'infer_progress':infer_progress, 
                  'single_path':single_path
                  }
        return {'result': result}
        