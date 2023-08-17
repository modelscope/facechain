# Copyright (c) Alibaba, Inc. and its affiliates.
import enum
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import gradio as gr
import numpy as np
import torch
from modelscope import snapshot_download

from facechain.inference import GenPortrait
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn

training_threadpool = ThreadPoolExecutor(max_workers=1)
inference_threadpool = ThreadPoolExecutor(max_workers=5)

training_done_count = 0
inference_done_count = 0

HOT_MODELS = [
    "\N{fire}数字身份",
]

examples = {
    'prompt_male': [
        ['wearing silver armor'],
        ['wearing T-shirt']
    ],
    'prompt_female': [
        ['wearing beautiful traditional hanfu, upper_body'],
        ['wearing an elegant evening gown']
    ],
}

example_styles = [
    {'name': '默认风格(default_style_model_path)'},
    {'name': '凤冠霞帔(Chinese traditional gorgeous suit)',
     'model_id': 'ly261666/civitai_xiapei_lora',
     'revision': 'v1.0.0',
     'bin_file': 'xiapei.safetensors',
     'multiplier_style': 0.35,
     'add_prompt_style': 'red, hanfu, tiara, crown, '},
]


class UploadTarget(enum.Enum):
    PERSONAL_PROFILE = 'Personal Profile'
    LORA_LIaBRARY = 'LoRA Library'


def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def train_lora_fn(foundation_model_path=None, revision=None, output_img_dir=None, work_dir=None):
    os.system(
        f'PYTHONPATH=. accelerate launch facechain/train_text_to_image_lora.py --pretrained_model_name_or_path={foundation_model_path} '
        f'--revision={revision} --sub_path="film/film" '
        f'--output_dataset_name={output_img_dir} --caption_column="text" --resolution=512 '
        f'--random_flip --train_batch_size=1 --num_train_epochs=200 --checkpointing_steps=5000 '
        f'--learning_rate=1e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 --seed=42 --output_dir={work_dir} '
        f'--lora_r=32 --lora_alpha=32 --lora_text_encoder_r=32 --lora_text_encoder_alpha=32')


def launch_pipeline(uuid,
                    prompt_cloth,
                    user_models,
                    num_images=1,
                    style_model=None,
                    ):
    base_model = 'ly261666/cv_portrait_model'
    before_queue_size = inference_threadpool._work_queue.qsize()
    before_done_count = inference_done_count
    multiplier_style = None
    add_prompt_style = None

    if style_model == example_styles[0]['name']:
        style_model_path = None
    else:
        style_model_path = style_model
        for e in example_styles:
            if style_model == e['name']:
                model_dir = snapshot_download(e['model_id'], revision=e['revision'])
                style_model_path = os.path.join(model_dir, e['bin_file'])
                multiplier_style = e['multiplier_style']
                add_prompt_style = e['add_prompt_style']

    print("-------user_models: ", user_models)
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            return "请登陆后使用! "
        else:
            uuid = 'qw'

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False

    output_model_name = 'personalizaition_lora'
    instance_data_dir = os.path.join('/tmp', uuid, 'training_data', output_model_name)

    lora_model_path = f'/tmp/{uuid}/{output_model_name}'

    gen_portrait = GenPortrait(prompt_cloth, style_model_path, multiplier_style, add_prompt_style, use_main_model,
                               use_face_swap, use_post_process,
                               use_stylization)

    num_images = min(6, num_images)
    future = inference_threadpool.submit(gen_portrait, instance_data_dir,
                                         num_images, base_model, lora_model_path, 'film/film', 'v2.0')

    while not future.done():
        is_processing = future.running()
        if not is_processing:
            cur_done_count = inference_done_count
            to_wait = before_queue_size - (cur_done_count - before_done_count)
            yield ["排队等待资源中，前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5), None]
        else:
            yield ["生成中, 请耐心等待...", None]
        time.sleep(1)

    outputs = future.result()
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))
    image_path = './lora_result.png'
    if len(outputs) > 0:
        result = concatenate_images(outputs)
        cv2.imwrite(image_path, result)

        yield ["生成完毕！", outputs_RGB]
    else:
        yield ["生成失败，请重试！", outputs_RGB]


class Trainer:
    def __init__(self):
        pass

    def run(
            self,
            uuid: str,
            instance_images: list,
    ) -> str:

        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')
        if instance_images is None:
            raise gr.Error('您需要上传训练图片！')
        if len(instance_images) > 10:
            raise gr.Error('您需要上传小于10张训练图片！')
        if not uuid:
            if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
                return "请登陆后使用! "
            else:
                uuid = 'qw'

        output_model_name = 'personalizaition_lora'

        # mv user upload data to target dir
        instance_data_dir = os.path.join('/tmp', uuid, 'training_data', output_model_name)
        print("--------uuid: ", uuid)

        if not os.path.exists(f"/tmp/{uuid}"):
            os.makedirs(f"/tmp/{uuid}")
        work_dir = f"/tmp/{uuid}/{output_model_name}"
        print("----------work_dir: ", work_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(instance_data_dir, ignore_errors=True)

        prepare_dataset([img['name'] for img in instance_images], output_dataset_dir=instance_data_dir)
        data_process_fn(instance_data_dir, True)

        # train lora
        train_lora_fn(foundation_model_path='ly261666/cv_portrait_model',
                      revision='v2.0',
                      output_img_dir=instance_data_dir,
                      work_dir=work_dir)

        message = f'训练已经完成！请切换至 [形象体验] 标签体验模型效果'
        print(message)
        return message


def flash_model_list(uuid):
    folder_path = f"/tmp/{uuid}"
    folder_list = []
    print("------flash_model_list folder_path: ", folder_path)
    if not os.path.exists(folder_path):
        print('--------The folder_path is missing.')
    else:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/output/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)

    print("-------folder_list + HOT_MODELS: ", folder_list + HOT_MODELS)
    return gr.Radio.update(choices=HOT_MODELS + folder_list)


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def train_input():
    trainer = Trainer()

    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('训练数据')
                    instance_images = gr.Gallery()
                    upload_button = gr.UploadButton("选择图片上传", file_types=["image"], file_count="multiple")
                    upload_button.upload(upload_file, upload_button, instance_images)
                    gr.Markdown('''
                        - Step 1. 上传你计划训练的图片，3~10张头肩照（注意：图片中多人脸、脸部遮挡等情况会导致效果异常，需要重新上传符合规范图片训练）
                        - Step 2. 点击 [形象定制] ，启动模型训练，等待约15分钟，请您耐心等待
                        - Step 3. 切换至 [形象体验] ，生成你的风格照片
                        ''')

        run_button = gr.Button('开始训练（等待上传图片加载显示出来再点，否则会报错）')

        with gr.Box():
            gr.Markdown(
                '输出信号（出现error时训练可能已完成或还在进行。可直接切到形象体验tab页面，如果体验时报错则训练还没好，再等待一般10来分钟。）')
            output_message = gr.Markdown()
        with gr.Box():
            gr.Markdown('''
            碰到抓狂的错误或者计算资源紧张的情况下，推荐直接在[NoteBook](https://modelscope.cn/my/mynotebook/preset)上进行体验，
            安装方法请参考：https://github.com/modelscope/facechain
            ''')

        run_button.click(fn=trainer.run,
                         inputs=[
                             uuid,
                             instance_images,
                         ],
                         outputs=[output_message])

    return demo


def inference_input():
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        with gr.Row():
            with gr.Column():
                user_models = gr.Radio(label="模型选择", choices=HOT_MODELS, type="value", value=HOT_MODELS[0])
                prompt_cloth = gr.Textbox(label="服饰相关提示词", value='wearing high-class business/working suit')
                gr.Examples(examples['prompt_male'], inputs=[prompt_cloth], label='男性提示词示例')
                gr.Examples(examples['prompt_female'], inputs=[prompt_cloth], label='女性提示词示例')
                style_model = gr.Textbox(label="风格模型选择(当不是默认风格时服饰相关提示词不生效)", value=example_styles[0]['name'])
                gr.Examples([e['name'] for e in example_styles], inputs=[style_model], label='风格模型列表')

                with gr.Box():
                    num_images = gr.Number(
                        label='生成图片数量', value=6, precision=1)
                    gr.Markdown('''
                    注意：最多支持生成6张图片!
                        ''')

        display_button = gr.Button('开始推理')

        with gr.Box():
            infer_progress = gr.Textbox(label="生成进度", value="当前无生成任务", interactive=False)
        with gr.Box():
            gr.Markdown('生成结果')
            output_images = gr.Gallery(label='Output', show_label=False).style(columns=3, rows=2, height=600,
                                                                               object_fit="contain")
        display_button.click(fn=launch_pipeline,
                             inputs=[uuid, prompt_cloth, user_models, num_images, style_model],
                             outputs=[infer_progress, output_images])

    return demo


with gr.Blocks(css='style.css') as demo:
    with gr.Tabs():
        with gr.TabItem('\N{rocket}形象定制'):
            train_input()
        with gr.TabItem('\N{party popper}形象体验'):
            inference_input()

demo.queue(status_update_rate=1).launch(share=True)
