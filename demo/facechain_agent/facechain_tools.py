from modelscope_agent.tools import Tool, TextToImageTool
import gradio as gr
from typing import List
import os
import json
from difflib import SequenceMatcher

#StyleSearchTool()
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

