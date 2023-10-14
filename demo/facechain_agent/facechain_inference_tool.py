import sys
sys.path.append("../../")
from modelscope_agent.tools import Tool
from facechain.utils import snapshot_download
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor
from torch import multiprocessing
import cv2
import numpy as np
import gradio as gr
from facechain.inference import GenPortrait

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
    character_model='ly261666/cv_portrait_model'#(base_model)
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
    # Check lora choice
    # if lora_choice == None:
    #     raise '请选择LoRA模型(Please select the LoRA model)!'
    # Check style model
    

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
    instance_data_dir = os.path.join('./', uuid, 'training_data', character_model, user_model)
    lora_model_path = f'./{uuid}/{character_model}/{user_model}/'
    #print('----------======================')
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

    save_dir = os.path.join('./', uuid, 'inference_result', base_model, user_model)
    if lora_choice == 'preset':
        save_dir = os.path.join(save_dir, 'style_' + style_model)
    else:
        save_dir = os.path.join(save_dir, 'lora_' + os.path.basename(lora_choice).split('.')[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # use single to save outputs
    if not os.path.exists(os.path.join(save_dir, 'single')):
        os.makedirs(os.path.join(save_dir, 'single'))
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

        return ("生成完毕(Generation done)!", outputs_RGB)
    else:
        return ("生成失败, 请重试(Generation failed, please retry)!", outputs_RGB)

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
    description = "模型微调推理，根据用户人脸Lora以及输入的期望风格输出图片"
    name = "facechain_inference_tool"
    parameters: list = [{
        'name': 'matched_style_file',
        'description': '用户输入的文本信息',
        'required': True
    }]

    def __init__(self,user_model:str):
        self.user_model = user_model
        super().__init__()
        # self.base_model_path = 'ly261666/cv_portrait_model'
        # self.revision = 'v2.0'
        # self.sub_path = "film/film"
        # # 这里固定了Lora的名字,重新训练会覆盖原来的
        # self.lora_name = "person1"

    def _remote_call(self, matched_style_file_path:str):
        with open (matched_style_file_path,'r') as f:
            matched_style_file = json.load(f)
        pos_prompt = generate_pos_prompt(matched_style_file, matched_style_file['add_prompt_style'])
        (infer_progress,output_images)=launch_pipeline(uuid='qw',matched=matched_style_file,pos_prompt=pos_prompt,
               neg_prompt=neg_prompt, base_model_index=0,
               user_model=self.user_model, num_images=3,
                multiplier_style=0.35,
               multiplier_human=0.95, pose_model=None,
               pose_image=None, lora_choice='preset'
               )
        return infer_progress,output_images

    def _local_call(self, matched_style_file_path:str):
        with open (matched_style_file_path,'r') as f:
            matched_style_file = json.load(f)
        pos_prompt = generate_pos_prompt(matched_style_file, matched_style_file['add_prompt_style'])
        (infer_progress,output_images)=launch_pipeline(uuid='qw',matched=matched_style_file,pos_prompt=pos_prompt,
               neg_prompt=neg_prompt, base_model_index=0,
               user_model=self.user_model, num_images=3,
                multiplier_style=0.35,
               multiplier_human=0.95, pose_model=None,
               pose_image=None, lora_choice='preset'
               )
        return infer_progress,output_images
        
with gr.Blocks() as demo:
    display_button = gr.Button('开始生成(Start!)')  
    with gr.Box():
            infer_progress = gr.Textbox(label="生成进度(Progress)", value="当前无生成任务(No task)", interactive=False)
    with gr.Box():
        gr.Markdown('生成结果(Result)')
        output_images = gr.Gallery(label='Output', show_label=False).style(columns=3, rows=2, height=600,object_fit="contain")
    tool = FaceChainInferenceTool(user_model='person1')
    with gr.Row():
        matched_style_file_path=gr.Text(value='/home/wsco/wyj2/facechain-agent/styles/leosamsMoonfilm_filmGrain20/Chinese_traditional_gorgeous_suit.json',)
    display_button.click(fn=tool._local_call,
                             inputs=[matched_style_file_path],
                             outputs=[infer_progress, output_images])        
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    demo.queue(status_update_rate=1).launch(share=True)