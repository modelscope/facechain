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
from facechain.facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
#from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from torch import multiprocessing


# class StyleSearchTool(Tool):
#     description = ''
#     name = 'style_search'
#     parameters: list = [{
#         'name': 'text',
#         'description': '用户输入的想要的风格文本',
#         'required': True
#     }]

#     def __init__(self, filepath: str, model_id: str ):
#         self.embeddings = ModelScopeEmbeddings(model_id=model_id) 
#         self.db = self.build_database(filepath)  # 使用FAISS构建文档嵌入向量数据库
#         #self.folder_path=folder_path
#         super().__init__()

#     def build_database(self,filepath):
#         docs=[]
#         loader = TextLoader(filepath, autodetect_encoding=True)
#         textsplitter = CharacterTextSplitter()
#         docs += (loader.load_and_split(textsplitter))
#         db = FAISS.from_documents(docs, self.embeddings)
#         return db

#     def _remote_call(self, text):
#         query_vector = self.embeddings.encode([text])[0]  # 将查询文本转换为嵌入向量
#         similar_doc_indices, _ = self.db.search([query_vector], k=1)  # 查找最相似的文档

#         # # 获取最相似的文档的name字段
#         # best_match_name = self.get_name_from_document_index(similar_doc_indices[0][0])
#         best_match_name = similar_doc_indices[0][0]
#         result = {
#             'name': self.name,
#             'best_match_name': best_match_name
#         }
#         return {'result': result}

#     def _local_call(self, text):
#         query_vector = self.embeddings.encode([text])[0]  # 将查询文本转换为嵌入向量
#         similar_doc_indices, _ = self.db.search([query_vector], k=1)  # 查找最相似的文档

#         # 获取最相似的文档的name字段
#         #best_match_name = self.get_name_from_document_index(similar_doc_indices[0][0])
#         best_match_name = similar_doc_indices[0][0]
#         result = {
#             'name': self.name,
#             'best_match_name': best_match_name
#         }
#         return {'result': result}
    
class StyleSearchTool(Tool):
    description = ''
    name = 'style_search'
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
   
