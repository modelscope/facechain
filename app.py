# Copyright (c) Alibaba, Inc. and its affiliates.
import enum
import os
import json
import shutil
import slugify
import time
import cv2
import gradio as gr
import numpy as np
import torch
from glob import glob
import platform
from PIL import Image
from importlib.util import find_spec
from facechain.inference_fact import GenPortrait
from facechain.inference_inpaint_fact import GenPortrait_inpaint
from facechain.utils import snapshot_download, check_ffmpeg, project_dir, join_worker_data_dir
from train_style.demo import set_img, init_tag, cut_img, train_lora, set_prompt
from facechain.constants import neg_prompt as neg, pos_prompt_with_cloth, pos_prompt_with_style, \
    pose_examples, base_models, tts_speakers_map


inference_done_count = 0
character_model = 'ly261666/cv_portrait_model'
BASE_MODEL_MAP = {
    "leosamsMoonfilm_filmGrain20": "写实模型(Realistic sd_1.5 model)",
    "MajicmixRealistic_v6": "\N{fire}写真模型(Photorealistic sd_1.5 model)",
}


class UploadTarget(enum.Enum):
    PERSONAL_PROFILE = 'Personal Profile'
    LORA_LIaBRARY = 'LoRA Library'


# utils
def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def select_function(evt: gr.SelectData):
    name = evt.value[1] if isinstance(evt.value, (tuple, list)) else evt.value
    matched = list(filter(lambda item: name == item['name'], styles))
    style = matched[0]
    return gr.Text.update(value=style['name'], visible=True)

def select_function_multi(evt: gr.SelectData):
    tag = evt.value[1]
    impath = evt.value[0]
    return gr.Text.update(value=impath), gr.Text.update(value=tag)

def get_selected_image(state_image_list, evt: gr.SelectData):
    return state_image_list[evt.index]

def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths


def update_prompt(style_model, style_choice, uuid):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    if style_choice == 0:
        matched = list(filter(lambda item: style_model == item['name'], styles))
        style = matched[0]
        pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'])
        multiplier_style = style['multiplier_style']
        multiplier_human = style['multiplier_human']
    else:
        f = open(f'{project_dir}/workspace/{uuid}/style_lora/{style_model}/add_prompt_style.txt', 'r')
        add_prompt_style = f.read()
        f.close()
        pos_prompt = pos_prompt_with_style.format(add_prompt_style)
        multiplier_style = 0.8
        
    return gr.Textbox.update(value=pos_prompt), \
           gr.Slider.update(value=multiplier_style)


def update_pose_model(pose_image, pose_model):
    if pose_image is None:
        return gr.Radio.update(value=pose_models[0]['name']), gr.Image.update(visible=False)
    else:
        if pose_model == 0:
            pose_model = 1
        pose_res_img = preprocess_pose(pose_image)
        return gr.Radio.update(value=pose_models[pose_model]['name']), gr.Image.update(value=pose_res_img, visible=True)



def generate_pos_prompt(style_model, prompt_cloth):
    if style_model is not None:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        if matched['model_id'] is None:
            pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
        else:
            pos_prompt = pos_prompt_with_style.format(matched['add_prompt_style'])
    else:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    return pos_prompt


def launch_pipeline(uuid,
                    style_choice,
                    pos_prompt,
                    neg_prompt=None,
                    user_images=None,
                    num_images=1,
                    style_model=None,
                    lora_choice=None,
                    multiplier_style=0.35,
                    pose_image=None,
                    use_face_swap=0
                    ):
    
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    
    # Check style model
    if style_choice == None:
        raise gr.Error('请选择风格模型(Please select the style model)!')
    
    if style_model == None and lora_choice == 'preset':
        raise gr.Error('请选择风格模型(Please select the style model)!')
    
    before_queue_size = 0
    before_done_count = inference_done_count
    
    if style_choice == 0:
        matched = list(filter(lambda item: style_model == item['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        style_model = matched['name']

    if lora_choice == 'preset':
        if style_choice == 1:
            style_model_path = os.path.join(f'{project_dir}/workspace/{uuid}/style_lora', style_model, 'lora_weights.safetensors')
            base_model_index = 0
        elif matched['model_id'] is None:
            style_model_path = None
            base_model_index = 0
        else:
            model_dir = snapshot_download(matched['model_id'], revision=matched['revision'])
            style_model_path = os.path.join(model_dir, matched['bin_file'])
            base_model_index = matched['base_model_index']
    else:
        print(f'uuid: {uuid}')
        temp_lora_dir = join_worker_data_dir(uuid, 'temp_lora')
        file_name = lora_choice
        print(lora_choice.split('.')[-1], os.path.join(temp_lora_dir, file_name))
        if lora_choice.split('.')[-1] != 'safetensors' or not os.path.exists(os.path.join(temp_lora_dir, file_name)):
            raise ValueError(f'Invalid lora file: {lora_file.name}')
        style_model_path = os.path.join(temp_lora_dir, file_name)
        base_model_index = 1

    num_images = min(6, num_images)
    print('base model index: ', base_model_index)
    
    outputs = gen_portrait(use_face_swap, num_images, base_model_index, style_model_path, pos_prompt, neg_prompt, user_images[0]['name'], pose_image, multiplier_style)

    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))
    
    if len(outputs) > 0:
        yield ["生成完毕(Generation done)!", outputs_RGB]
    else:
        yield ["生成失败, 请重试(Generation failed, please retry)!", outputs_RGB]


def launch_pipeline_inpaint(uuid,
                            user_images,
                            num_faces=1,
                            selected_face=1,
                            template_image=None,
                            use_face_swap=0):

    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    
#    if isinstance(user_image, str):
#        if len(user_image) == 0:
#            raise gr.Error('请选择一张用户图像(Please select 1 user image)')

    if isinstance(template_image, str):
        if len(template_image) == 0:
            raise gr.Error('请选择一张模板(Please select 1 template)')

    multiplier_style = 0.05
    strength = 0.6
    output_img_size = 512

    pos_prompt = 'raw photo, masterpiece, simple background, solo, medium shot, high detail face, photorealistic, best quality, wearing T-shirt'
    neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
                'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'

    outputs = gen_portrait_inpaint(use_face_swap, template_image,
                 strength,
                 output_img_size,
                 num_faces,
                 selected_face,
                 pos_prompt,
                 neg_prompt,
                 user_images[0]['name'])
    
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))

    if len(outputs) > 0:
        yield ["生成完毕(Generation done)！", outputs_RGB]
    else:
        yield ["生成失败，请重试(Generation failed, please retry)！", outputs_RGB]

        
def update_lora_choice(uuid):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    print("uuid: ", uuid)
    temp_lora_dir = join_worker_data_dir(uuid, 'temp_lora')
    if not os.path.exists(temp_lora_dir):
        os.makedirs(temp_lora_dir)
    
    lora_list = sorted(os.listdir(temp_lora_dir))
    lora_list = ["preset"] + lora_list
    
    return gr.Dropdown.update(choices=lora_list, value="preset")

def upload_lora_file(uuid, lora_file):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    print("uuid: ", uuid)
    temp_lora_dir = join_worker_data_dir(uuid, 'temp_lora')
    if not os.path.exists(temp_lora_dir):
        os.makedirs(temp_lora_dir)
    shutil.copy(lora_file.name, temp_lora_dir)
    filename = os.path.basename(lora_file.name)
    newfilepath = os.path.join(temp_lora_dir, filename)
    print("newfilepath: ", newfilepath)
    
    lora_list = sorted(os.listdir(temp_lora_dir))
    lora_list = ["preset"] + lora_list
    
    return gr.Dropdown.update(choices=lora_list, value=filename)


def clear_lora_file(uuid, lora_file):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    
    return gr.Dropdown.update(value="preset")


def change_lora_choice(lora_choice):
    
    if lora_choice == 'preset':
        return gr.Gallery.update(value=[(item["img"], item["name"]) for item in styles], visible=True), \
               gr.Text.update(value=style_list[0])
    else:
        return gr.Gallery.update(visible=False), gr.Text.update(visible=False)


def change_style_choice(uuid, style_choice):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    out_path = f'{project_dir}/workspace/{uuid}/style_lora'
    if os.path.exists(out_path):
        choices = os.listdir(out_path)
    else:
        choices = []
    
    if style_choice == 0:
        return gr.Gallery.update(visible=True), gr.Radio.update(choices=choices, visible=False)
    else:
        return gr.Gallery.update(visible=False), gr.Radio.update(choices=choices, visible=True)

def select_trained_style(trained_styles):
    return gr.Text.update(value=trained_styles)

def get_tag(imgs):
    results = []
    for i in range(len(imgs)):
        file, old_prompt = imgs[i]
        img_path = file['name']
        img = Image.open(img_path)
        result = tag_model.tag(img, threshold=0.7)
        results.append([img_path, result])
        imgs[i][1] = result

    return gr.Gallery.update(value=results, visible=True)

def modify_tag(gallery, impath, tag):
    results = []
    for item in gallery:
        if item[0]['data'] == impath:
            results.append([item[0]['name'], tag])
        else:
            results.append([item[0]['name'], item[1]])
    return gr.Gallery.update(value=results)
    
def inference_input():
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    style_choice = gr.Radio(label="风格模型来源(Whether enhancing face similarity)", choices=["预设风格(Preset styles)", "用户训练风格(User-trained styles)"], type="index", value=None)
                    style_model = gr.Text(label='请选择一种风格(Select a style from the pics below):', interactive=False)
                    trained_styles = gr.Radio(label='用户训练风格列表(User-trained style list)', choices=[], value=None, type="value", visible=False)
                    
                    if find_spec('webui'):
                        gallery = gr.Gallery(value=[(item["img"], item["name"]) for item in styles],
                                            label="风格(Style)",
                                            allow_preview=False,
                                            elem_id="gallery",
                                            show_share_button=False,
                                            visible=True).style(columns=6, rows=2)
                    else:
                        gallery = gr.Gallery(value=[(item["img"], item["name"]) for item in styles],
                                            label="风格(Style)",
                                            allow_preview=False,
                                            elem_id="gallery",
                                            show_share_button=False,
                                            visible=True).style(columns=6, object_fit='contain', height=600)
                
                with gr.Box():
                    gr.Markdown('请上传一张用户人像图片(Please upload a user image)：')
                    user_images = gr.Gallery(label="输入用户图片(User image)", show_label=True)
            
                    with gr.Row(elem_id="container_row"):
                        upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                                    file_count="multiple")
                        clear_button = gr.Button("清空图片(Clear photos)")

                    clear_button.click(fn=lambda: [], inputs=None, outputs=user_images)
                    upload_button.upload(upload_file, inputs=[upload_button, user_images], outputs=user_images,
                                                     queue=False)

                with gr.Accordion("高级选项(Advanced Options)", open=False):
                    # upload one lora file and show the name or path of the file
                    with gr.Accordion("上传LoRA文件(Upload LoRA file)", open=False):
                        with gr.Row():
                            lora_choice = gr.Dropdown(choices=["preset"], type="value", value="preset", label="LoRA文件(LoRA file)", visible=True)
                            update_button = gr.Button('刷新风格LoRA列表并切换为预设风格(Refresh style LoRAs and switch to preset styles)')
                            
                        lora_file = gr.File(
                            value=None,
                            label="上传LoRA文件(Upload LoRA file)",
                            type="file",
                            file_types=[".safetensors"],
                            file_count="single",
                            visible=True,
                        )
                        
                    pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3,
                                            value=generate_pos_prompt(None, styles[0]['add_prompt_style']),
                                            interactive=True)
                    neg_prompt = gr.Textbox(label="负向提示语(Negative Prompt)", lines=3,
                                            value="",
                                            interactive=True)
                    if neg_prompt.value == '' :
                        neg_prompt.value = neg
                    multiplier_style = gr.Slider(minimum=0, maximum=1, value=0.25,
                                                 step=0.05, label='风格权重(Multiplier style)')
                    
                    with gr.Accordion("姿态控制(Pose control)", open=True):
                        with gr.Row():
                            pose_image = gr.Image(source='upload', type='filepath', label='姿态图片(Pose image)', height=250)
                            pose_res_image = gr.Image(source='upload', interactive=False, label='姿态结果(Pose result)', visible=False, height=250)
                        gr.Examples(pose_examples['man'], inputs=[pose_image], label='男性姿态示例')
                        gr.Examples(pose_examples['woman'], inputs=[pose_image], label='女性姿态示例')

                with gr.Box():
                    num_images = gr.Number(
                        label='生成图片数量(Number of photos)', value=1, precision=1, minimum=1, maximum=6)
                    use_face_swap = gr.Radio(label="是否使用人脸相似度增强(Whether enhancing face similarity)", choices=["否(No)", "是(Yes)"], type="index", value="是(Yes)")
                    gr.Markdown('''
                    注意:
                    - 最多支持生成6张图片!(You may generate a maximum of 6 photos at one time!)
                    - 可上传在定义LoRA文件使用, 否则默认使用风格模型的LoRA。(You may upload custome LoRA file, otherwise the LoRA file of the style model will be used by deault.)
                    - 使用自定义LoRA文件需手动输入prompt, 否则可能无法正常触发LoRA文件风格。(You shall provide prompt when using custom LoRA, otherwise desired LoRA style may not be triggered.)
                        ''')

        with gr.Row(elem_id="container_row"):
            display_button = gr.Button('开始生成(Start!)', variant='primary')

        with gr.Box():
            infer_progress = gr.Textbox(label="生成进度(Progress)", value="当前无生成任务(No task)", interactive=False)
        with gr.Box():
            gr.Markdown('生成结果(Result)')
            output_images = gr.Gallery(label='Output', show_label=False).style(columns=3, rows=2, height=600,
                                                                               object_fit="contain")
        
        style_choice.change(fn=change_style_choice, inputs=[uuid, style_choice], outputs=[gallery, trained_styles], queue=False)
        gallery.select(select_function, None, style_model, queue=False)
        trained_styles.change(select_trained_style, inputs=[trained_styles], outputs=[style_model], queue=False)
        
        lora_choice.change(fn=change_lora_choice, inputs=[lora_choice], outputs=[gallery, style_model], queue=False)
        
        lora_file.upload(fn=upload_lora_file, inputs=[uuid, lora_file], outputs=[lora_choice], queue=False)
        lora_file.clear(fn=clear_lora_file, inputs=[uuid, lora_file], outputs=[lora_choice], queue=False)
        
        style_model.change(update_prompt, [style_model, style_choice, uuid], [pos_prompt, multiplier_style], queue=False)
        
        display_button.click(fn=launch_pipeline,
                             inputs=[uuid, style_choice, pos_prompt, neg_prompt, user_images, num_images, style_model, lora_choice, multiplier_style,
                                     pose_image, use_face_swap],
                             outputs=[infer_progress, output_images])
        
        update_button.click(fn=update_lora_choice, inputs=[uuid], outputs=[lora_choice], queue=False)

    return demo


def inference_inpaint():
    preset_template = glob(os.path.join(f'{project_dir}/inpaint_template/*.jpg'))
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        # Initialize the GUI

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('请选择或上传模板图片(Please select or upload a template image)：')
                    template_image_list = [[i] for idx, i in enumerate(preset_template)]
                    print(template_image_list)
                    template_image = gr.Image(source='upload', type='filepath', label='模板图片(Template image)')
                    gr.Examples(template_image_list, inputs=[template_image], label='模板示例(Template examples)')
                
                with gr.Box():
                    gr.Markdown('请上传用户人像图片(Please upload a user image)：')
                    user_images = gr.Gallery(label="输入用户图片(User image)", show_label=True)
            
                    with gr.Row(elem_id="container_row"):
                        upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                                    file_count="multiple")
                        clear_button = gr.Button("清空图片(Clear photos)")

                    clear_button.click(fn=lambda: [], inputs=None, outputs=user_images)
                    upload_button.upload(upload_file, inputs=[upload_button, user_images], outputs=user_images,
                                                     queue=False)
                    
                num_faces = gr.Number(minimum=1, value=1, precision=1, label='照片中的人脸数目(Number of Faces)')
                selected_face = gr.Number(minimum=1, value=1, precision=1, label='选择重绘的人脸编号，按从左至右的顺序(Index of Face for inpainting, counting from left to right)')
                use_face_swap = gr.Radio(label="是否使用人脸相似度增强(Whether enhancing face similarity)", choices=["否(No)", "是(Yes)"], type="index", value="是(Yes)")

        with gr.Row(elem_id="container_row"):
            display_button = gr.Button('开始生成(Start Generation)', variant='primary')
        with gr.Box():
            infer_progress = gr.Textbox(
                label="生成(Generation Progress)",
                value="No task currently",
                interactive=False
            )
        with gr.Box():
            gr.Markdown('生成结果(Generated Results)')
            output_images = gr.Gallery(
                label='输出(Output)',
                show_label=False
            ).style(columns=3, rows=2, height=600, object_fit="contain")

        display_button.click(
            fn=launch_pipeline_inpaint,
            inputs=[uuid, user_images, num_faces, selected_face, template_image, use_face_swap],
            outputs=[infer_progress, output_images]
        )

    return demo


def train_input():
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        
        output_model_name = gr.Text(label='风格lora模型名称(Style lora name)', visible=True)
        
        gallery = gr.Gallery(type='image', label='图片列表(Photos)', height=250, columns=8, visible=True)
        with gr.Row(elem_id="container_row"):
            upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"], file_count="multiple")
        train_folder = gr.Text(label='训练文件夹(Train folder)', visible=False)

        with gr.Row(elem_id="container_row"):
            tag_btn = gr.Button(value='开始打标签(Tag prompt)')
            rank = gr.Number(label='rank', direction='row', value=32, step=1)
            num_train_epochs = gr.Number(label='num_train_epochs', direction='row', value=200, step=1)
        
        with gr.Accordion("手动修改标签(Manually modify tags)", open=False):
            with gr.Row(elem_id="container_row"):
                current_pth = gr.Text(label='当前图片(Current image)', value=None, visible=False)
                current_tag = gr.Text(label='当前标签(Current tags)', value=None, visible=True)
                mod_btn = gr.Button(value='提交修改(Submit modifications)')

        prompt_input = gr.Text(label='风格触发词(Trigger word)', visible=True)
        with gr.Row(elem_id="container_row"):
            btn = gr.Button(value='开始训练(Start train)', interactive=False)
        output_lora = gr.Files(label='输出模型(Output model)', type='file', visible=True)
        output_prompt = gr.Text(label='风格提示词(Style prompt)', visible=False)

        # 完成待训练图片上传
        upload_button.upload(fn=set_img, inputs=[upload_button, uuid, output_model_name], outputs=[train_folder, gallery, btn])
        # 完成公用提示词输入
        prompt_input.input(fn=set_prompt, outputs=[btn])
        # 开始给图片打标签（prompt）
        tag_btn.click(fn=get_tag, inputs=[gallery], outputs=[gallery])
        # 获取图片标签
        gallery.select(select_function_multi, None, [current_pth, current_tag], queue=False)
        # 手动修改标签
        mod_btn.click(fn=modify_tag, inputs=[gallery, current_pth, current_tag], outputs=[gallery], queue=False)
        # 开始训练
        btn.click(fn=train_lora, inputs=[uuid, output_model_name, prompt_input, train_folder, gallery, rank, num_train_epochs], outputs=[output_lora, output_prompt])
    return demo


styles = []
style_list = []
base_models_reverse = [base_models[1], base_models[0]]
for base_model in base_models_reverse:
    folder_path = f"{os.path.dirname(os.path.abspath(__file__))}/styles/{base_model['name']}"
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            if data['img'][:2] == './':
                data['img'] = f"{project_dir}/{data['img'][2:]}"
                if base_model['name'] == 'leosamsMoonfilm_filmGrain20':
                    data['base_model_index'] = 0
                else:
                    data['base_model_index'] = 1
            style_list.append(data['name'])
            styles.append(data)

for style in styles:
    print(style['name'])
    if style['model_id'] is not None:
        model_dir = snapshot_download(style['model_id'], revision=style['revision'])

gen_portrait = GenPortrait()
gen_portrait_inpaint = GenPortrait_inpaint()
tag_model = init_tag()

with open(
        os.path.join(os.path.dirname(__file__), 'main.css'), "r",
        encoding="utf-8") as f:
    MAIN_CSS_CODE = f.read()

with gr.Blocks(css=MAIN_CSS_CODE, theme=gr.themes.Soft()) as demo:
    if find_spec('webui'):
        # if running as a webui extension, don't display banner self-advertisement
        gr.Markdown("# <center> \N{fire} FaceChain-FACT Portrait Generation (\N{whale} [Github star it here](https://github.com/modelscope/facechain/tree/main) \N{whale})</center>")
    else:
        gr.Markdown("# <center> \N{fire} FaceChain-FACT Portrait Generation ([Github star it here](https://github.com/modelscope/facechain/tree/main) \N{whale},   [API](https://help.aliyun.com/zh/dashscope/developer-reference/facechain-quick-start) \N{whale})</center>")
    gr.Markdown("##### <center> 本项目仅供学习交流，请勿将模型及其制作内容用于非法活动或违反他人隐私的场景。(This project is intended solely for the purpose of technological discussion, and should not be used for illegal activities and violating privacy of individuals.)</center>")
    with gr.Tabs():
        with gr.TabItem('\N{party popper}免训练无限风格形象写真(Infinite Style Portrait)'):
            inference_input()
        with gr.TabItem('\N{party popper}免训练固定模板形象写真(Fixed Templates Portrait)'):
            inference_inpaint()
        with gr.TabItem('\N{party popper}自定义风格模型训练(Style Model Training)'):
            train_input()

if __name__ == "__main__":
    demo.queue(status_update_rate=1).launch(share=False)
