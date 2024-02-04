# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor
import cv2
import gradio as gr
import numpy as np
import torch
import multiprocessing as mp
from diffusers import StableDiffusionPipeline, PNDMScheduler
from face_adapter import FaceAdapter, Face_Extracter
from face_adapter.merge_lora import merge_lora

from modelscope import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image

inference_done_count = 0
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

def set_spawn_method():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("spawn method already set")

def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths

def select_function(evt: gr.SelectData):
    name = evt.value[1] if isinstance(evt.value, (tuple, list)) else evt.value
    matched = list(filter(lambda item: name == item['name'], styles))
    style = matched[0]
    return gr.Text.update(value=style['name'], visible=True)



def gen_portrait(base_model_path, style_model_path, multiplier_style, pos_prompt, neg_prompt, input_img, face_adapter_scale, cfg_scale, num_images):
    # pre & post process model
    segmentation_pipeline = pipeline(
        Tasks.image_segmentation,
        'damo/cv_resnet101_image-multiple-human-parsing',
        model_revision='v1.0.1')
    
    face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_resnet50_face-detection_retinaface')
    fact_model_dir = snapshot_download('iic/face_chain_fact_model', revision='v1.0.0')
    face_adapter_path = os.path.join(fact_model_dir, 'face_adapter/adapter_maj_25.ckpt')
    # face_adapter_path = './model/adapter_maj_25.ckpt' 需更改这里的路径指向自己训练的fact参数
    face_extracter = Face_Extracter(fr_weight_path=os.path.join(fact_model_dir, 'face_adapter/ms1mv2_model_TransFace_S.pt'), \
                                    fc_weight_path=os.path.join(fact_model_dir, 'face_adapter/adapter_maj_25.ckpt'))

    pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            safety_checker=None,
            torch_dtype=torch.float16)

    pipe.scheduler = PNDMScheduler.from_config(
                    pipe.scheduler.config)
    pipe = merge_lora(
            pipe,
            style_model_path,
            multiplier_style,
            device='cuda',
            from_safetensor=True)

    face_adapter = FaceAdapter(pipe, face_detection, segmentation_pipeline, face_extracter, face_adapter_path, 'cuda')
    face_adapter.set_scale(face_adapter_scale)
    outputs = []
    batch_size = 1
    for i in range(int(num_images / batch_size)):
        images_style = face_adapter.generate(prompt=pos_prompt, face_image=input_img, height=512, width=512, 
            guidance_scale=cfg_scale, negative_prompt=neg_prompt, num_inference_steps=50, num_images_per_prompt=batch_size)
        if (np.array(face_detection(np.array(images_style[0]))['scores']) > 0.5).sum() ==1:
            outputs.append(np.array(images_style[0]))
    return outputs

def launch_pipeline(uuid,
                    instance_images=None,
                    num_images=1,
                    style_model=None
                    ):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'

    if instance_images == None or len(instance_images) != 1:
        raise gr.Error('请上传一张图片(Please upload an image)!')

    if style_model == None:
        raise gr.Error('请选择风格模型(Please select the style model)!')
    
    before_queue_size = 0
    before_done_count = inference_done_count
    # load reference image
    input_img_name = instance_images[0]['name']
    input_img = Image.open(input_img_name).convert('RGB')
    w, h = input_img.size
    if max(w, h) > 1000:
        scale = 1000 / max(w, h)
        input_img = input_img.resize((int(w * scale), int(h * scale)))


    # hyper-parater for style lora
    matched = list(filter(lambda item: style_model == item['name'], styles))
    if len(matched) == 0:
        raise ValueError(f'styles not found: {style_model}')
    style_info = matched[0]
    # print(style_info)
    multiplier_style = 0.8
    multiplier_style = style_info['multiplier_style']
    style_model_id = style_info['model_id']
    style_revision = style_info['revision']
    style_file = style_info['bin_file']
    style_model_dir = snapshot_download(style_model_id, revision=style_revision)
    style_model_path = os.path.join(style_model_dir, style_file)
    add_prompt_style = style_info['add_prompt_style']
    # add_prompt_style = "1male, face shot, Denim jackets, West style, checkered shirts, leather vests, yellow wide-brimmed hats, bandanas, leather belts with silver buckles, short hair, tough, desert background"
    # print(add_prompt_style)
    # import pdb
    # pdb.set_trace()

    pos_prompt = add_prompt_style + ', upper_body, raw photo, masterpiece, solo, medium shot, high detail face, photorealistic, best quality'
    neg_prompt = ''

    # hyper-parater for FaceChain-Fact
    face_adapter_scale = 0.7
    cfg_scale = 5.0

    # base model & load model + face adapter
    base_model_path = 'YorickHe/majicmixRealistic_v6'
    base_model_revision = 'v1.0.0'
    base_sub_path = 'realistic'

    base_model_path = snapshot_download(base_model_path, revision=base_model_revision)
    if base_sub_path is not None and len(base_sub_path) > 0:
        base_model_path = os.path.join(base_model_path, base_sub_path)

    num_images = min(6, num_images)

    with ProcessPoolExecutor(max_workers=5) as executor:
        future = executor.submit(gen_portrait, base_model_path, style_model_path, multiplier_style, pos_prompt, neg_prompt, input_img, face_adapter_scale, cfg_scale, num_images)
        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = inference_done_count
                to_wait = before_queue_size - (cur_done_count - before_done_count)
                yield ["排队等待资源中, 前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5),
                        None]
            else:
                yield ["生成中, 请耐心等待(Generating)...", None]
            time.sleep(1)

    outputs = future.result()

    if len(outputs) > 0:
        yield ["生成完毕(Generation done)!", outputs]
    else:
        yield ["生成失败, 请重试(Generation failed, please retry)!", outputs]


def inference_input():
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        
        with gr.Row():
            with gr.Column():
                instance_images = gr.Gallery()
                with gr.Row():
                    upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                    file_count="multiple")

                upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images,
                                        queue=False)

                with gr.Box():
                    style_model = gr.Text(label='请选择一种风格(Select a style from the pics below):', interactive=False)
                    gallery = gr.Gallery(value=[(item["img"], item["name"]) for item in styles],
                                        label="风格(Style)",
                                        allow_preview=True,
                                        columns=5,
                                        elem_id="gallery",
                                        show_share_button=False,
                                        visible=True)
                with gr.Box():
                    num_images = gr.Number(
                        label='生成图片数量(Number of photos)', value=1, precision=1, minimum=1, maximum=6)
                    gr.Markdown('''
                    注意: 
                    - 最多支持生成6张图片!(You may generate a maximum of 6 photos at one time!)
                        ''')

        with gr.Row():
            display_button = gr.Button('开始生成(Start!)')   

        with gr.Box():
            infer_progress = gr.Textbox(label="生成进度(Progress)", value="当前无生成任务(No task)", interactive=False)
        with gr.Box():
            gr.Markdown('生成结果(Result)')
            output_images = gr.Gallery(label='Output', show_label=False).style(columns=3, rows=2, height=600,
                                                                               object_fit="contain")
        
        gallery.select(select_function, None, style_model, queue=False)
        display_button.click(fn=launch_pipeline,
                        inputs=[uuid, instance_images, num_images, style_model],
                        outputs=[infer_progress, output_images])
    return demo


styles = []
folder_path = f"{os.path.dirname(os.path.abspath(__file__))}/styles"
files = os.listdir(folder_path)
files.sort()
for file in files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        if data['img'][:2] == './':
            data['img'] = f"{os.path.dirname(os.path.abspath(__file__))}/{data['img'][2:]}"
        styles.append(data)

with gr.Blocks(css='style.css') as demo:
    from importlib.util import find_spec
    gr.Markdown("# <center> \N{fire} FaceChain Fact Zero-shot Generation ([Github star it here](https://github.com/modelscope/facechain/tree/main) \N{whale},   [Paper](https://arxiv.org/abs/2308.14256) \N{whale})</center>")
    gr.Markdown("##### <center> 本项目仅供学习交流，请勿将模型及其制作内容用于非法活动或违反他人隐私的场景。(This project is intended solely for the purpose of technological discussion, and should not be used for illegal activities and violating privacy of individuals.)</center>")
    with gr.Tabs():
        with gr.TabItem('\N{party popper}FaceChain Fact 形象写真(FaceChain Fact Portrait)'):
            inference_input()

if __name__ == "__main__":
    set_spawn_method()
    demo.queue(status_update_rate=1).launch(share=True)
