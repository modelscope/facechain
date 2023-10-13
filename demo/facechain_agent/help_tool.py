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
#from facechain.facechain.train_text_to_image_lora import
from facechain.inference import data_process_fn
#from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from torch import multiprocessing
from PIL import Image
# class StyleSearchTool(Tool):
#     description = '搜索风格类型'
#     name = 'style_search'
#     parameters: list = [{
#         'name': 'text',
#         'description': '用户输入的想要的风格文本',
#         'required': True
#     }]

#     def __init__(self, stylepath: List[str], model_id: List[str]):
#         self.embeddings = ModelScopeEmbeddings(model_id=model_id)  # 加载预训练的文本嵌入模型
#         self.db = self.build_database(stylepath)  # 使用FAISS构建文档嵌入向量数据库
#         #self.folder_path=folder_path
#         super().__init__()

#     def build_database(self,stylepath):
#         list_of_documents = []
#         # 获取指定文件夹下的所有 JSON 文件
#         json_files = list(Path(stylepath).glob("*.json"))
#         # 遍历每个 JSON 文件并读取内容
#         for json_file in json_files:
#             with open(json_file, "r", encoding="utf-8") as file:
#                 json_data = json.load(file)             
#                 # 获取 JSON 中的name字段
#                 content = json_data.get("name", "")              
#                 # 创建 Document 对象并添加到列表中
#                 document = Document(page_content=content)
#                 list_of_documents.append(document)
#         # 使用FAISS.from_documents构建文档嵌入向量数据库
#         db = FAISS.from_documents(list_of_documents, self.embeddings)
#         # db.save_local("faiss_index")
#         #db = FAISS.load_local("faiss_index", self.embeddings)
#         #new_db = FAISS.load_local("faiss_index", embeddings)
#         return db
    
#     def _local_call(self, text):
#         docs = self.db.similarity_search(text,k=1)
#         result = {
#             'name': self.name,
#             'best_match_name': docs[0].page_content
#         }
#         return {'result': result}
    
#     def _remote_call(self, text):
#         docs = self.db.similarity_search(text,k=1)
#         result = {
#             'name': self.name,
#             'best_match_name': docs[0].page_content
#         }
#         return {'result': result}
    
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

      # 列出风格文件夹下的所有文件
      files = os.listdir(self.style_path)
      files.sort()

      for file in files:
          file_path = os.path.join(self.style_path, file)
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

        # 列出风格文件夹下的所有文件
        files = os.listdir(self.style_path)
        files.sort()

        for file in files:
            file_path = os.path.join(self.style_path, file)
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

    def __init__(self):
        super().__init__()
        self.base_model_path = 'ly261666/cv_portrait_model'
        self.revision = 'v2.0'
        self.sub_path = "film/film"
        # 这里固定了Lora的名字,重新训练会覆盖原来的
        self.lora_name = "person1"
        

    def _remote_call(self):
        uuid = 'qw'
        #multiprocessing.set_start_method('spawn')
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
    max_train_steps = min(photo_num * 200, 800)

    if platform.system() == 'Windows':
        command = [
            'accelerate', 'launch', 'facechain/train_text_to_image_lora.py',
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
            f'PYTHONPATH=. accelerate launch /home/wsco/wyj2/facechain-agent/facechain/train_text_to_image_lora.py '
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
    instance_data_dir = os.path.join('./', uuid, 'training_data', base_model_path, output_model_name)
    print(instance_data_dir)
    if not os.path.exists(f"./{uuid}"):
        os.makedirs(f"./{uuid}")
    work_dir = f"./{uuid}/{base_model_path}/{output_model_name}"

    # if os.path.exists(work_dir):
    #     raise gr.Error("人物lora名称已存在。(This character lora name already exists.)")

    shutil.rmtree(work_dir, ignore_errors=True)
    
    #shutil.rmtree(instance_data_dir, ignore_errors=True)
    #prepare_dataset([img['name'] for img in instance_images], output_dataset_dir=instance_data_dir)
    #data_process_fn(instance_data_dir, True)

    train_lora_fn(
        base_model_path=base_model_path,
        revision=revision,
        sub_path=sub_path,
        output_img_dir=instance_data_dir,
        work_dir=work_dir,
        photo_num=len(instance_data_dir)
    )

    return base_model_path, revision, sub_path, instance_data_dir, work_dir


# 生成uuid
def generate_id():
    timestamp = str(int(time.time()))
    random_num = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    return timestamp + random_num