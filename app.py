# Copyright (c) Alibaba, Inc. and its affiliates.
import enum
import os
import shutil
import slugify
import time
from concurrent.futures import ProcessPoolExecutor
import uuid as UUID
from typing import Tuple
from torch import multiprocessing
import cv2
import gradio as gr
import numpy as np
import torch
from glob import glob
import platform
import subprocess
from facechain.utils import snapshot_download

from facechain.inference import preprocess_pose, GenPortrait
from facechain.inference_inpaint import GenPortrait_inpaint
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, styles, \
    pose_models, pose_examples, base_models

training_done_count = 0
inference_done_count = 0

is_spaces = True if "SPACE_ID" in os.environ else False
if is_spaces:
    is_shared_ui = True if "modelscope/FaceChain" in os.environ['SPACE_ID'] else False
else:
    is_shared_ui = False
is_gpu_associated = torch.cuda.is_available()


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
    matched = list(filter(lambda item: evt.value == item['name'], styles))
    style = matched[0]
    return gr.Text.update(value=style['name'], visible=True)


def update_prompt(style_model):
    matched = list(filter(lambda item: style_model == item['name'], styles))
    style = matched[0]
    pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'])
    multiplier_style = style['multiplier_style']
    multiplier_human = style['multiplier_human']
    return gr.Textbox.update(value=pos_prompt), \
        gr.Slider.update(value=multiplier_style), \
        gr.Slider.update(value=multiplier_human)


def update_pose_model(pose_image, pose_model):
    if pose_image is None:
        return gr.Radio.update(value=pose_models[0]['name']), gr.Image.update(visible=False)
    else:
        if pose_model == 0:
            pose_model = 1
        pose_res_img = preprocess_pose(pose_image)
        return gr.Radio.update(value=pose_models[pose_model]['name']), gr.Image.update(value=pose_res_img, visible=True)


def update_optional_styles(base_model_index):
    style_list = base_models[base_model_index]['style_list']
    optional_styles = '\n'.join(style_list)
    return gr.Textbox.update(value=optional_styles)


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
    else:
        os.system(
            f'PYTHONPATH=. accelerate launch facechain/train_text_to_image_lora.py '
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


def generate_pos_prompt(style_model, prompt_cloth):
    if style_model in base_models[0]['style_list'][:-1] or style_model is None:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    else:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        pos_prompt = pos_prompt_with_style.format(matched['add_prompt_style'])
    return pos_prompt


def launch_pipeline(uuid,
                    pos_prompt,
                    neg_prompt=None,
                    base_model_index=None,
                    user_model=None,
                    num_images=1,
                    lora_choice=None,
                    style_model=None,
                    multiplier_style=0.25,
                    multiplier_human=0.85,
                    pose_model=None,
                    pose_image=None
                    ):
    if not uuid:
        uuid = UUID.uuid4().hex

    # Check base model
    if base_model_index == None:
        raise gr.Error('请选择基模型(Please select the base model)!')

    # Check character LoRA
    base_model_path = base_models[base_model_index]['model_id']
    folder_path = f"/tmp/{uuid}/{base_model_path}"
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
        raise gr.Error(
            '该基模型下没有人物LoRA，请先训练(There is no character LoRA under this base model, please train first)!')

    # Check output model
    if user_model == None:
        raise gr.Error('请选择人物LoRA(Please select the character LoRA)！')
    # Check lora choice
    if lora_choice == None:
        raise gr.Error('请选择LoRA模型(Please select the LoRA model)!')
    # Check style model
    if style_model == None and lora_choice == 'preset':
        raise gr.Error('请选择风格模型(Please select the style model)!')

    base_model = base_models[base_model_index]['model_id']
    revision = base_models[base_model_index]['revision']
    sub_path = base_models[base_model_index]['sub_path']

    before_queue_size = 0
    before_done_count = inference_done_count
    matched = list(filter(lambda item: style_model == item['name'], styles))
    style_model = matched[0]['name']

    if lora_choice == 'preset':
        if style_model in base_models[0]['style_list'][:-1]:
            style_model_path = None
        else:
            matched = list(filter(lambda style: style_model == style['name'], styles))
            if len(matched) == 0:
                raise ValueError(f'styles not found: {style_model}')
            matched = matched[0]
            model_dir = snapshot_download(matched['model_id'], revision=matched['revision'])
            style_model_path = os.path.join(model_dir, matched['bin_file'])
    else:
        print(f'uuid: {uuid}')
        temp_lora_dir = f"/tmp/{uuid}/temp_lora"
        file_name = lora_choice
        print(lora_choice.split('.')[-1], os.path.join(temp_lora_dir, file_name))
        if lora_choice.split('.')[-1] != 'safetensors' or not os.path.exists(os.path.join(temp_lora_dir, file_name)):
            raise ValueError(f'Invalid lora file: {lora_file.name}')
        style_model_path = os.path.join(temp_lora_dir, file_name)

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

    print("-------user_model: ", user_model)
    if not uuid:
        uuid = UUID.uuid4().hex

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False

    instance_data_dir = os.path.join('/tmp', uuid, 'training_data', base_model, user_model)
    lora_model_path = f'/tmp/{uuid}/{base_model}/{user_model}/ensemble'
    if not os.path.exists(lora_model_path):
        lora_model_path = f'/tmp/{uuid}/{base_model}/{user_model}/'

    gen_portrait = GenPortrait(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path,
                               multiplier_style, multiplier_human, use_main_model,
                               use_face_swap, use_post_process,
                               use_stylization)

    num_images = min(6, num_images)

    with ProcessPoolExecutor(max_workers=5) as executor:
        future = executor.submit(gen_portrait, instance_data_dir,
                                 num_images, base_model, lora_model_path, sub_path, revision)
        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = inference_done_count
                to_wait = before_queue_size - (cur_done_count - before_done_count)
                yield [uuid,
                       "排队等待资源中, 前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5),
                       None]
            else:
                yield [uuid, "生成中, 请耐心等待(Generating)...", None]
            time.sleep(1)

    outputs = future.result()
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))

    save_dir = os.path.join('/tmp', uuid, 'inference_result', base_model, user_model)
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
        cv2.imwrite(image_path, result)

        yield [uuid, "生成完毕(Generation done)!", outputs_RGB]
    else:
        yield [uuid, "生成失败, 请重试(Generation failed, please retry)!", outputs_RGB]


def launch_pipeline_inpaint(uuid,
                            base_model_index=None,
                            user_model_A=None,
                            user_model_B=None,
                            num_faces=1,
                            template_image=None):
    before_queue_size = 0
    before_done_count = inference_done_count

    if not uuid:
        uuid = UUID.uuid4().hex

    # Check base model
    if base_model_index == None:
        raise gr.Error('请选择基模型(Please select the base model)！')

    # Check character LoRA
    base_model_path = base_models[base_model_index]['model_id']
    folder_path = f"/tmp/{uuid}/{base_model_path}"
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
        raise gr.Error(
            '该基模型下没有人物LoRA，请先训练(There is no character LoRA under this base model, please train first)!')

    # Check character LoRA
    if num_faces == 1:
        if user_model_A == None:
            raise gr.Error('请至少选择一个人物LoRA(Please select at least one character LoRA)！')
    else:
        if user_model_A == None and user_model_B == None:
            raise gr.Error('请至少选择一个人物LoRA(Please select at least one character LoRA)！')

    if not uuid:
        uuid = UUID.uuid4().hex

    if isinstance(template_image, str):
        if len(template_image) == 0:
            raise gr.Error('请选择一张模板(Please select 1 template)')

    base_model = base_models[base_model_index]['model_id']
    revision = base_models[base_model_index]['revision']
    sub_path = base_models[base_model_index]['sub_path']
    multiplier_style = 0.05
    multiplier_human = 0.95
    strength = 0.65
    output_img_size = 512

    model_dir = snapshot_download('ly261666/cv_wanx_style_model', revision='v1.0.3')
    style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_frommajicreal.safetensors')

    pos_prompt = 'raw photo, masterpiece, chinese, simple background, high-class pure color background, solo, medium shot, high detail face, photorealistic, best quality, wearing T-shirt'
    neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
                 'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'

    if user_model_A == '不重绘该人物(Do not inpaint this character)':
        user_model_A = None
    if user_model_B == '不重绘该人物(Do not inpaint this character)':
        user_model_B = None

    if user_model_A is not None:
        instance_data_dir_A = os.path.join('/tmp', uuid, 'training_data', base_model, user_model_A)
        lora_model_path_A = f'/tmp/{uuid}/{base_model}/{user_model_A}/'
    else:
        instance_data_dir_A = None
        lora_model_path_A = None
    if user_model_B is not None:
        instance_data_dir_B = os.path.join('/tmp', uuid, 'training_data', base_model, user_model_B)
        lora_model_path_B = f'/tmp/{uuid}/{base_model}/{user_model_B}/'
    else:
        instance_data_dir_B = None
        lora_model_path_B = None

    in_path = template_image
    out_path = 'inpaint_rst'

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False

    gen_portrait = GenPortrait_inpaint(in_path, strength, num_faces,
                                       pos_prompt, neg_prompt, style_model_path,
                                       multiplier_style, multiplier_human, use_main_model,
                                       use_face_swap, use_post_process,
                                       use_stylization)

    with ProcessPoolExecutor(max_workers=5) as executor:
        future = executor.submit(gen_portrait, instance_data_dir_A, instance_data_dir_B, base_model, \
                                 lora_model_path_A, lora_model_path_B, sub_path=sub_path, revision=revision)

        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = inference_done_count
                to_wait = before_queue_size - (cur_done_count - before_done_count)
                yield [uuid,
                       "排队等待资源中，前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5),
                       None]
            else:
                yield [uuid, "生成中, 请耐心等待(Generating)...", None]
            time.sleep(1)

    outputs = future.result()
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))

    for i, out_tmp in enumerate(outputs):
        cv2.imwrite('{}_{}.png'.format(out_path, i), out_tmp)

    if len(outputs) > 0:
        yield [uuid, "生成完毕(Generation done)！", outputs_RGB]
    else:
        yield [uuid, "生成失败，请重试(Generation failed, please retry)！", outputs_RGB]


class Trainer:
    def __init__(self):
        pass

    def run(
            self,
            uuid: str,
            instance_images: list,
            base_model_index: int,
            output_model_name: str,
    ) -> Tuple[str, str]:
        # Check Cuda
        if not torch.cuda.is_available():
            raise gr.Error('CUDA不可用(CUDA not available)')

        # Check Instance Valid
        if instance_images is None:
            raise gr.Error('您需要上传训练图片(Please upload photos)!')

        # Check output model name
        if not output_model_name:
            raise gr.Error('请指定人物lora的名称(Please specify the character LoRA name)！')

        # Limit input Image
        if len(instance_images) > 20:
            raise gr.Error('请最多上传20张训练图片(20 images at most!)')

        # Check UUID & Studio
        if not uuid:
            uuid = UUID.uuid4().hex

        base_model_path = base_models[base_model_index]['model_id']
        revision = base_models[base_model_index]['revision']
        sub_path = base_models[base_model_index]['sub_path']
        output_model_name = slugify.slugify(output_model_name)

        # mv user upload data to target dir
        instance_data_dir = os.path.join('/tmp', uuid, 'training_data', base_model_path, output_model_name)
        print("--------uuid: ", uuid)

        if not os.path.exists(f"/tmp/{uuid}"):
            os.makedirs(f"/tmp/{uuid}")
        work_dir = f"/tmp/{uuid}/{base_model_path}/{output_model_name}"

        if os.path.exists(work_dir):
            raise gr.Error("人物lora名称已存在。(This character lora name already exists.)")

        print("----------work_dir: ", work_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(instance_data_dir, ignore_errors=True)

        prepare_dataset([img['name'] for img in instance_images], output_dataset_dir=instance_data_dir)
        data_process_fn(instance_data_dir, True)

        # train lora
        print("instance_data_dir", instance_data_dir)
        train_lora_fn(base_model_path=base_model_path,
                      revision=revision,
                      sub_path=sub_path,
                      output_img_dir=instance_data_dir,
                      work_dir=work_dir,
                      photo_num=len(instance_images))

        message = '''<center><font size=4>训练已经完成！请切换至 [无限风格形象写真] 标签体验模型效果。</center>
        
        <center><font size=4>(Training done, please switch to the Infinite Style Portrait tab to generate photos.)</center>'''
        print(message)
        return uuid, message


def flash_model_list(uuid, base_model_index, lora_choice: gr.Dropdown):
    base_model_path = base_models[base_model_index]['model_id']
    style_list = base_models[base_model_index]['style_list']

    sub_styles = []
    for style in style_list:
        matched = list(filter(lambda item: style == item['name'], styles))
        sub_styles.append(matched[0])

    if not uuid:
        uuid = UUID.uuid4().hex

    folder_path = f"/tmp/{uuid}/{base_model_path}"
    folder_list = []
    lora_save_path = f"/tmp/{uuid}/temp_lora"
    if not os.path.exists(lora_save_path):
        lora_list = ['preset']
    else:
        lora_list = sorted(os.listdir(lora_save_path))
        lora_list = ["preset"] + lora_list

    if not os.path.exists(folder_path):
        if lora_choice == 'preset':
            return uuid, gr.Radio.update(choices=[]), \
                gr.Gallery.update(value=[(item["img"], item["name"]) for item in sub_styles], visible=True), \
                gr.Text.update(value=style_list[0], visible=True), \
                gr.Dropdown.update(choices=lora_list, visible=True), gr.File.update(visible=True)
        else:
            return uuid, gr.Radio.update(choices=[]), \
                gr.Gallery.update(visible=False), gr.Text.update(), \
                gr.Dropdown.update(choices=lora_list, visible=True), gr.File.update(visible=True)
    else:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)

    if lora_choice == 'preset':
        return uuid, gr.Radio.update(choices=folder_list), \
            gr.Gallery.update(value=[(item["img"], item["name"]) for item in sub_styles], visible=True), \
            gr.Text.update(value=style_list[0], visible=True), \
            gr.Dropdown.update(choices=lora_list, visible=True), gr.File.update(visible=True)
    else:
        return uuid, gr.Radio.update(choices=folder_list), \
            gr.Gallery.update(visible=False), gr.Text.update(), \
            gr.Dropdown.update(choices=lora_list, visible=True), gr.File.update(visible=True)


def update_output_model(uuid, base_model_index):
    # Check base model
    if base_model_index == None:
        raise gr.Error('请选择基模型(Please select the base model)!')

    base_model_path = base_models[base_model_index]['model_id']
    style_list = base_models[base_model_index]['style_list']

    if not uuid:
        uuid = UUID.uuid4().hex

    folder_path = f"/tmp/{uuid}/{base_model_path}"
    folder_list = []
    if not os.path.exists(folder_path):
        return uuid, gr.Radio.update(choices=[]), gr.Dropdown.update(choices=style_list)
    else:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)

    return uuid, gr.Radio.update(choices=folder_list)


def update_output_model_inpaint(uuid, base_model_index):
    # Check base model
    if base_model_index == None:
        raise gr.Error('请选择基模型(Please select the base model)！')

    base_model_path = base_models[base_model_index]['model_id']
    style_list = base_models[base_model_index]['style_list']

    if not uuid:
        uuid = UUID.uuid4().hex

    folder_path = f"/tmp/{uuid}/{base_model_path}"
    folder_list = ['不重绘该人物(Do not inpaint this character)']
    if not os.path.exists(folder_path):
        return uuid, gr.Radio.update(choices=[]), gr.Dropdown.update(choices=style_list)
    else:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)

    return uuid, gr.Radio.update(choices=folder_list, value=folder_list[0]), gr.Radio.update(choices=folder_list,
                                                                                       value=folder_list[0])


def update_output_model_num(num_faces):
    if num_faces == 1:
        return uuid, gr.Radio.update(), gr.Radio.update(visible=False)
    else:
        return uuid, gr.Radio.update(), gr.Radio.update(visible=True)


def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths


def upload_lora_file(uuid, lora_file):
    if not uuid:
        uuid = UUID.uuid4().hex
    print("uuid: ", uuid)
    temp_lora_dir = f"/tmp/{uuid}/temp_lora"
    if not os.path.exists(temp_lora_dir):
        os.makedirs(temp_lora_dir)
    shutil.copy(lora_file.name, temp_lora_dir)
    filename = os.path.basename(lora_file.name)
    newfilepath = os.path.join(temp_lora_dir, filename)
    print("newfilepath: ", newfilepath)

    lora_list = sorted(os.listdir(temp_lora_dir))
    lora_list = ["preset"] + lora_list

    return uuid, gr.Dropdown.update(choices=lora_list, value=filename)


def clear_lora_file(uuid, lora_file):
    if not uuid:
        uuid = UUID.uuid4().hex

    return uuid, gr.Dropdown.update(value="preset")


def change_lora_choice(lora_choice, base_model_index):
    style_list = base_models[base_model_index]['style_list']
    sub_styles = []
    for style in style_list:
        matched = list(filter(lambda item: style == item['name'], styles))
        sub_styles.append(matched[0])

    if lora_choice == 'preset':
        return gr.Gallery.update(value=[(item["img"], item["name"]) for item in sub_styles], visible=True), \
            gr.Text.update(value=style_list[0])
    else:
        return gr.Gallery.update(visible=False), gr.Text.update(visible=False)


def deal_history(uuid, base_model_index=None, user_model=None, lora_choice=None, style_model=None, deal_type="load"):
    if not uuid:
        uuid = UUID.uuid4().hex

    if base_model_index is None:
        raise gr.Error('请选择基模型(Please select the base model)!')
    if user_model is None:
        raise gr.Error('请选择人物lora(Please select the character lora)!')
    if lora_choice is None:
        raise gr.Error('请选择LoRa文件(Please select the LoRa file)!')
    if style_model is None and lora_choice == 'preset':
        raise gr.Error('请选择风格(Please select the style)!')

    base_model = base_models[base_model_index]['model_id']
    matched = list(filter(lambda item: style_model == item['name'], styles))
    style_model = matched[0]['name']

    save_dir = os.path.join('/tmp', uuid, 'inference_result', base_model, user_model)
    if lora_choice == 'preset':
        save_dir = os.path.join(save_dir, 'style_' + style_model)
    else:
        save_dir = os.path.join(save_dir, 'lora_' + os.path.basename(lora_choice).split('.')[0])

    if not os.path.exists(save_dir):
        return uuid, gr.Gallery.update(value=[], visible=True), gr.Gallery.update(value=[], visible=True)

    if deal_type == "load":
        single_dir = os.path.join(save_dir, 'single')
        concat_dir = os.path.join(save_dir, 'concat')
        single_imgs = []
        concat_imgs = []
        if os.path.exists(single_dir):
            single_imgs = sorted(os.listdir(single_dir))
            single_imgs = [os.path.join(single_dir, img) for img in single_imgs]
        if os.path.exists(concat_dir):
            concat_imgs = sorted(os.listdir(concat_dir))
            concat_imgs = [os.path.join(concat_dir, img) for img in concat_imgs]

        return uuid, gr.Gallery.update(value=single_imgs, visible=True), gr.Gallery.update(value=concat_imgs, visible=True)
    elif deal_type == "delete":
        shutil.rmtree(save_dir)
        return uuid, gr.Gallery.update(value=[], visible=True), gr.Gallery.update(value=[], visible=True)


def train_input(uuid):
    trainer = Trainer()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('模型选择(Model list)')

                    base_model_list = []
                    for base_model in base_models:
                        base_model_list.append(base_model['name'])

                    base_model_index = gr.Radio(label="基模型选择(Base model list)", choices=base_model_list,
                                                type="index",
                                                value=base_model_list[0])

                    optional_style = '\n'.join(base_models[0]['style_list'])

                    optional_styles = gr.Textbox(label="该基模型支持的风格(Styles supported by this base model.)",
                                                 max_lines=5,
                                                 value=optional_style, interactive=False)

                    output_model_name = gr.Textbox(label="人物lora名称(Character lora name)", value='person1', lines=1)

                    gr.Markdown('训练图片(Training photos)')
                    instance_images = gr.Gallery()
                    upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                    file_count="multiple")

                    clear_button = gr.Button("清空图片(Clear photos)")
                    clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                    upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images,
                                         queue=False)

                    gr.Markdown('''
                        使用说明（Instructions）：
                        ''')
                    gr.Markdown('''
                        - Step 1. 上传计划训练的图片, 1~10张头肩照(注意: 请避免图片中出现多人脸、脸部遮挡等情况, 否则可能导致效果异常)
                        - Step 2. 点击 [开始训练] , 启动形象定制化训练, 每张图片约需1.5分钟, 请耐心等待～
                        - Step 3. 切换至 [形象写真] , 生成你的风格照片<br/><br/>
                        ''')
                    gr.Markdown('''
                        - Step 1. Upload 1-10 headshot photos of yours (Note: avoid photos with multiple faces or face obstruction, which may lead to non-ideal result).
                        - Step 2. Click [Train] to start training for customizing your Digital-Twin, this may take up-to 1.5 mins per image.
                        - Step 3. Switch to [Portrait] Tab to generate stylized photos.
                        ''')

        run_button = gr.Button('开始训练(等待上传图片加载显示出来再点, 否则会报错)... '
                               'Start training (please wait until photo(s) fully uploaded, otherwise it may result in training failure)')

        with gr.Box():
            gr.Markdown('''
            <center>请等待训练完成，请勿刷新或关闭页面。</center>

            <center>(Please wait for the training to complete, do not refresh or close the page.)</center>
            ''')
            output_message = gr.Markdown()
        with gr.Box():
            gr.Markdown('''
            碰到抓狂的错误或者计算资源紧张的情况下，推荐直接在[NoteBook](https://modelscope.cn/my/mynotebook/preset)上进行体验。

            (If you are experiencing prolonged waiting time, you may try on [ModelScope NoteBook](https://modelscope.cn/my/mynotebook/preset) to prepare your dedicated environment.)

            安装方法请参考：https://github.com/modelscope/facechain .

            (You may refer to: https://github.com/modelscope/facechain for installation instruction.)
            ''')
        base_model_index.change(fn=update_optional_styles,
                                inputs=[base_model_index],
                                outputs=[optional_styles],
                                queue=False)

        run_button.click(fn=trainer.run,
                         inputs=[
                             uuid,
                             instance_images,
                             base_model_index,
                             output_model_name,
                         ],
                         outputs=[uuid, output_message])

    return demo


def inference_input(uuid):
    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column():
                base_model_list = []
                for base_model in base_models:
                    base_model_list.append(base_model['name'])

                base_model_index = gr.Radio(label="基模型选择(Base model list)", choices=base_model_list, type="index")

                with gr.Row():
                    with gr.Column(scale=2):
                        user_model = gr.Radio(label="人物LoRA列表(Character LoRAs)", choices=[], type="value")
                    with gr.Column(scale=1):
                        update_button = gr.Button('刷新人物LoRA列表(Refresh character LoRAs)')

                with gr.Box():
                    style_model = gr.Text(label='请选择一种风格(Select a style from the pics below):',
                                          interactive=False)
                    gallery = gr.Gallery(value=[(item["img"], item["name"]) for item in styles],
                                         label="风格(Style)",
                                         allow_preview=False,
                                         columns=5,
                                         elem_id="gallery",
                                         show_share_button=False,
                                         visible=False)

                pmodels = []
                for pmodel in pose_models:
                    pmodels.append(pmodel['name'])

                with gr.Accordion("高级选项(Advanced Options)", open=False):
                    # upload one lora file and show the name or path of the file
                    with gr.Accordion("上传LoRA文件(Upload LoRA file)", open=False):
                        lora_choice = gr.Dropdown(choices=["preset"], type="value", value="preset",
                                                  label="LoRA文件(LoRA file)", visible=False)
                        lora_file = gr.File(
                            value=None,
                            label="上传LoRA文件(Upload LoRA file)",
                            type="file",
                            file_types=[".safetensors"],
                            file_count="single",
                            visible=False,
                        )

                    pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3,
                                            value=generate_pos_prompt(None, styles[0]['add_prompt_style']),
                                            interactive=True)
                    neg_prompt = gr.Textbox(label="负向提示语(Negative Prompt)", lines=3,
                                            value="",
                                            interactive=True)
                    multiplier_style = gr.Slider(minimum=0, maximum=1, value=0.25,
                                                 step=0.05, label='风格权重(Multiplier style)')
                    multiplier_human = gr.Slider(minimum=0, maximum=1.2, value=0.95,
                                                 step=0.05, label='形象权重(Multiplier human)')

                    with gr.Accordion("姿态控制(Pose control)", open=False):
                        with gr.Row():
                            pose_image = gr.Image(source='upload', type='filepath', label='姿态图片(Pose image)',
                                                  height=250)
                            pose_res_image = gr.Image(source='upload', interactive=False, label='姿态结果(Pose result)',
                                                      visible=False, height=250)
                        gr.Examples(pose_examples['man'], inputs=[pose_image], label='男性姿态示例')
                        gr.Examples(pose_examples['woman'], inputs=[pose_image], label='女性姿态示例')
                        pose_model = gr.Radio(choices=pmodels, value=pose_models[0]['name'],
                                              type="index", label="姿态控制模型(Pose control model)")
                with gr.Box():
                    num_images = gr.Number(
                        label='生成图片数量(Number of photos)', value=6, precision=1, minimum=1, maximum=6)
                    gr.Markdown('''
                    注意: 
                    - 最多支持生成6张图片!(You may generate a maximum of 6 photos at one time!)
                    - 可上传在定义LoRA文件使用, 否则默认使用风格模型的LoRA。(You may upload custome LoRA file, otherwise the LoRA file of the style model will be used by deault.)
                    - 使用自定义LoRA文件需手动输入prompt, 否则可能无法正常触发LoRA文件风格。(You shall provide prompt when using custom LoRA, otherwise desired LoRA style may not be triggered.)
                        ''')

        with gr.Row():
            display_button = gr.Button('开始生成(Start!)')
            with gr.Column():
                history_button = gr.Button('查看历史(Show history)')
                load_history_text = gr.Text("load", visible=False)
                delete_history_button = gr.Button('删除历史(Delete history)')
                delete_history_text = gr.Text("delete", visible=False)

        with gr.Box():
            infer_progress = gr.Textbox(label="生成进度(Progress)", value="当前无生成任务(No task)", interactive=False)
        with gr.Box():
            gr.Markdown('生成结果(Result)')
            output_images = gr.Gallery(label='Output', show_label=False).style(columns=3, rows=2, height=600,
                                                                               object_fit="contain")

        with gr.Accordion(label="历史生成结果(History)", open=False):
            with gr.Row():
                single_history = gr.Gallery(label='单张图片(Single image history)')
                batch_history = gr.Gallery(label='图片组(Batch image history)')

        gallery.select(select_function, None, style_model, queue=False)
        lora_choice.change(fn=change_lora_choice, inputs=[lora_choice, base_model_index],
                           outputs=[gallery, style_model], queue=False)

        lora_file.upload(fn=upload_lora_file, inputs=[uuid, lora_file], outputs=[lora_choice], queue=False)
        lora_file.clear(fn=clear_lora_file, inputs=[uuid, lora_file], outputs=[lora_choice], queue=False)

        style_model.change(update_prompt, style_model, [pos_prompt, multiplier_style, multiplier_human], queue=False)
        pose_image.change(update_pose_model, [pose_image, pose_model], [pose_model, pose_res_image])
        base_model_index.change(fn=flash_model_list,
                                inputs=[uuid, base_model_index, lora_choice],
                                outputs=[uuid, user_model, gallery, style_model, lora_choice, lora_file],
                                queue=False)
        update_button.click(fn=update_output_model,
                            inputs=[uuid, base_model_index],
                            outputs=[uuid, user_model],
                            queue=False)
        display_button.click(fn=launch_pipeline,
                             inputs=[uuid, pos_prompt, neg_prompt, base_model_index, user_model, num_images,
                                     lora_choice, style_model, multiplier_style, multiplier_human,
                                     pose_model, pose_image],
                             outputs=[uuid, infer_progress, output_images])
        history_button.click(fn=deal_history,
                             inputs=[uuid, base_model_index, user_model, lora_choice, style_model, load_history_text],
                             outputs=[uuid, single_history, batch_history])
        delete_history_button.click(fn=deal_history,
                                    inputs=[uuid, base_model_index, user_model, lora_choice, style_model,
                                            delete_history_text],
                                    outputs=[uuid, single_history, batch_history])

    return demo


def inference_inpaint(uuid):
    preset_template = glob(os.path.join('resources/inpaint_template/*.jpg'))
    with gr.Blocks() as demo:
        # Initialize the GUI

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('请选择或上传模板图片(Please select or upload a template image)：')
                    template_image_list = [[i] for idx, i in enumerate(preset_template)]
                    print(template_image_list)
                    template_image = gr.Image(source='upload', type='filepath', label='模板图片(Template image)')
                    gr.Examples(template_image_list, inputs=[template_image], label='模板示例(Template examples)')

                base_model_list = []
                for base_model in base_models:
                    base_model_list.append(base_model['name'])

                base_model_index = gr.Radio(
                    label="基模型选择(Base model list)",
                    choices=base_model_list,
                    type="index"
                )

                num_faces = gr.Number(minimum=1, maximum=2, value=1, precision=1,
                                      label='照片中的人脸数目(Number of Faces)')
                with gr.Row():
                    with gr.Column(scale=2):
                        user_model_A = gr.Radio(
                            label="第1个人物LoRA，按从左至右的顺序（1st Character LoRA，counting from left to right）",
                            choices=[], type="value")
                        user_model_B = gr.Radio(
                            label="第2个人物LoRA，按从左至右的顺序（2nd Character LoRA，counting from left to right）",
                            choices=[], type="value", visible=False)
                    with gr.Column(scale=1):
                        update_button = gr.Button('刷新人物LoRA列表(Refresh character LoRAs)')

        display_button = gr.Button('开始生成(Start Generation)')
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

        base_model_index.change(fn=update_output_model_inpaint,
                                inputs=[uuid, base_model_index],
                                outputs=[uuid, user_model_A, user_model_B],
                                queue=False)

        update_button.click(fn=update_output_model_inpaint,
                            inputs=[uuid, base_model_index],
                            outputs=[uuid, user_model_A, user_model_B],
                            queue=False)

        num_faces.change(fn=update_output_model_num,
                         inputs=[uuid, num_faces],
                         outputs=[uuid, user_model_A, user_model_B],
                         queue=False)

        display_button.click(
            fn=launch_pipeline_inpaint,
            inputs=[uuid, base_model_index, user_model_A, user_model_B, num_faces, template_image],
            outputs=[uuid, infer_progress, output_images]
        )

    return demo


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(
        "# <center> \N{fire} FaceChain Potrait Generation ([Github star it here](https://github.com/modelscope/facechain/tree/main) \N{whale}, [Paper cite it here](https://arxiv.org/abs/2308.14256) \N{whale})</center>")
    with gr.Box():
        if is_shared_ui:
            top_description = gr.HTML(f'''
                <div class="gr-prose" style="max-width: 80%">
                <p>If the waiting queue is too long, you can either run locally or duplicate the Space and run it on your own profile using a (paid) private A10G-large GPU for training. A A10G-large costs US$3.15/h. &nbsp;&nbsp;<a class="duplicate-button" style="display:inline-block" target="_blank" href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></p>
                <img style="position: absolute; top: 0;right: 0; height: 100%;margin-top: 0px !important" src="file=duplicate.png"> 
                </div>
            ''')
        elif is_spaces:
            if is_gpu_associated:
                top_description = gr.HTML(f'''
                                <div class="gr-prose" style="max-width: 80%">
                                <h2>You have successfully associated a GPU to the FaceChain Space 🎉</h2>
                                <p>You can now train your model! You will be billed by the minute from when you activated the GPU until when it is turned it off.</p> 
                                </div>
                        ''')
            else:
                top_description = gr.HTML(f'''
                                <div class="gr-prose" style="max-width: 80%">
                                <h2>You have successfully duplicated the FaceChain Space 🎉</h2>
                                <p>There's only one step left before you can train your model: <a href="https://huggingface.co/spaces/{os.environ['SPACE_ID']}/settings" style="text-decoration: underline" target="_blank">attribute a <b>A10G-large GPU</b> to it (via the Settings tab)</a> and run the training below. You will be billed by the minute from when you activate the GPU until when it is turned it off.</p> 
                                </div>
                        ''')
        else:
            top_description = gr.HTML(f'''
                            <div class="gr-prose" style="max-width: 80%">
                            <h2>You have successfully cloned the FaceChain Space locally 🎉</h2>
                            <p>Do a <code>pip install requirements.txt</code></p> 
                            </div>
                        ''')
    uuid = gr.State([])
    with gr.Tabs():
        with gr.TabItem('\N{rocket}人物形象训练(Train Digital Twin)'):
            train_input(uuid)
        with gr.TabItem('\N{party popper}无限风格形象写真(Infinite Style Portrait)'):
            inference_input(uuid)
        with gr.TabItem('\N{party popper}固定模板形象写真(Fixed Templates Portrait)'):
            inference_inpaint(uuid)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    demo.queue(status_update_rate=1).launch(share=True)
