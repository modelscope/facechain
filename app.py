# Copyright (c) Alibaba, Inc. and its affiliates.
import enum
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import uuid as UUID
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from modelscope import snapshot_download

from facechain.inference import GenPortrait
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, styles, cloth_prompt

training_threadpool = ThreadPoolExecutor(max_workers=1)
inference_threadpool = ThreadPoolExecutor(max_workers=5)

training_done_count = 0
inference_done_count = 0

HOT_MODELS = [
    "\N{fire}数字身份(Digital Identity)",
]

is_spaces = True if "SPACE_ID" in os.environ else False
if is_spaces:
    is_shared_ui = True if "modelscope/FaceChain" in os.environ['SPACE_ID'] else False
else:
    is_shared_ui = False
is_gpu_associated = torch.cuda.is_available()


class UploadTarget(enum.Enum):
    PERSONAL_PROFILE = 'Personal Profile'
    LORA_LIaBRARY = 'LoRA Library'

def update_cloth(style_index):
    prompts = []
    if style_index == 0:
        example_prompt = generate_pos_prompt(styles[style_index]['name'],
                                             cloth_prompt[0]['prompt'])
        for prompt in cloth_prompt:
            prompts.append(prompt['name'])
    else:
        example_prompt = generate_pos_prompt(styles[style_index]['name'],
                                             styles[style_index]['add_prompt_style'])
        prompts.append(styles[style_index]['cloth_name'])
    return gr.Radio.update(choices=prompts, value=prompts[0]), gr.Textbox.update(value=example_prompt)


def update_prompt(style_index, cloth_index):
    if style_index == 0:
        pos_prompt = generate_pos_prompt(styles[style_index]['name'],
                                         cloth_prompt[cloth_index]['prompt'])
    else:
        pos_prompt = generate_pos_prompt(styles[style_index]['name'],
                                         styles[style_index]['add_prompt_style'])
    return gr.Textbox.update(value=pos_prompt)

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


def generate_pos_prompt(style_model, prompt_cloth):
    if style_model == styles[0]['name'] or style_model is None:
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
                    user_models,
                    num_images=1,
                    style_model=None,
                    multiplier_style=0.25
                    ):
    base_model = 'ly261666/cv_portrait_model'
    before_queue_size = inference_threadpool._work_queue.qsize()
    before_done_count = inference_done_count
    style_model = styles[style_model]['name']

    if style_model == styles[0]['name']:
        style_model_path = None
    else:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        model_dir = snapshot_download(matched['model_id'], revision=matched['revision'])
        style_model_path = os.path.join(model_dir, matched['bin_file'])

    print("-------user_models: ", user_models)
    if not uuid:
        uuid = UUID.uuid4().hex

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False

    output_model_name = 'personalizaition_lora'
    instance_data_dir = os.path.join('/tmp', uuid, 'training_data', output_model_name)

    lora_model_path = f'/tmp/{uuid}/{output_model_name}'

    gen_portrait = GenPortrait(pos_prompt, neg_prompt, style_model_path, multiplier_style, use_main_model,
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
            yield [uuid, "排队等待资源中，前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5), None]
        else:
            yield [uuid, "生成中, 请耐心等待(Generating)...", None]
        time.sleep(1)

    outputs = future.result()
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))
    image_path = './lora_result.png'
    if len(outputs) > 0:
        result = concatenate_images(outputs)
        cv2.imwrite(image_path, result)

        yield [uuid, "生成完毕(Generating done)！", outputs_RGB]
    else:
        yield [uuid, "生成失败，请重试(Generating failed, please retry)！", outputs_RGB]


class Trainer:
    def __init__(self):
        pass

    def run(
            self,
            uuid: str,
            instance_images: list,
    ) -> Tuple[str, str]:

        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')
        if instance_images is None:
            raise gr.Error('您需要上传训练图片(Please upload photos)！')
        if len(instance_images) > 10:
            raise gr.Error('您需要上传小于10张训练图片(Please upload at most 10 photos)！')
        if not uuid:
            uuid = UUID.uuid4().hex

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

        message = f'训练已经完成！请切换至 [形象体验] 标签体验模型效果(Training done, please switch to the inference tab to generate photos.)'
        print(message)
        return uuid, message


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


def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths


def train_input(uuid):
    trainer = Trainer()
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('训练图片(Training photos)')
                    instance_images = gr.Gallery()
                    upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                    file_count="multiple", queue=False)

                    clear_button = gr.Button("清空图片(Clear photos)")
                    clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                    upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images, queue=False)
                    gr.Markdown('''
                        - Step 1. 上传计划训练的图片，3~10张头肩照（注意：请避免图片中出现多人脸、脸部遮挡等情况，否则可能导致效果异常）
                        - Step 2. 点击 [开始训练] ，启动形象定制化训练，约需15分钟，请耐心等待～
                        - Step 3. 切换至 [形象体验] ，生成你的风格照片
                        ''')
                    gr.Markdown('''
                        - Step 1. Upload 3-10 headshot photos of yours (Note: avoid photos with multiple faces or face obstruction, which may lead to non-ideal result).
                        - Step 2. Click [Train] to start training for customizing your Digital-Twin, this may take up-to 15 mins.
                        - Step 3. Switch to [Inference] Tab to generate stylized photos.
                        ''')

        run_button = gr.Button('开始训练（等待上传图片加载显示出来再点，否则会报错）'
                               'Start training (please wait until photo(s) fully uploaded, otherwise it may result in training failure)')

        with gr.Box():
            gr.Markdown('''
            请等待训练完成
            
            Please wait for the training to complete.
            ''')
            output_message = gr.Markdown()
        with gr.Box():
            gr.Markdown('''
            碰到抓狂的错误或者计算资源紧张的情况下，推荐直接在[NoteBook](https://modelscope.cn/my/mynotebook/preset)上进行体验。
            
            安装方法请参考：https://github.com/modelscope/facechain .
            
            If you are experiencing prolonged waiting time, you may try on [ModelScope NoteBook](https://modelscope.cn/my/mynotebook/preset) to prepare your dedicated environment.
                        
            You may refer to: https://github.com/modelscope/facechain for installation instruction.
            ''')

        run_button.click(fn=trainer.run,
                         inputs=[
                             uuid,
                             instance_images,
                         ],
                         outputs=[uuid, output_message])

    return demo


def inference_input(uuid):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                user_models = gr.Radio(label="模型选择(Model list)", choices=HOT_MODELS, type="value",
                                       value=HOT_MODELS[0])
                style_model_list = []
                for style in styles:
                    style_model_list.append(style['name'])
                style_model = gr.Dropdown(choices=style_model_list, value=styles[0]['name'], 
                                          type="index", label="风格模型(Style model)")
                
                prompts=[]
                for prompt in cloth_prompt:
                    prompts.append(prompt['name'])
                cloth_style = gr.Radio(choices=prompts, value=cloth_prompt[0]['name'],
                                       type="index", label="服装风格(Cloth style)")

                with gr.Accordion("高级选项(Expert)", open=False):
                    pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3,
                                        value=generate_pos_prompt(None, cloth_prompt[0]['prompt']), interactive=True)
                    multiplier_style = gr.Slider(minimum=0, maximum=1, value=0.25,
                                                 step=0.05, label='风格权重(Multiplier style)')
                with gr.Box():
                    num_images = gr.Number(
                        label='生成图片数量(Number of photos)', value=6, precision=1, minimum=1, maximum=6)
                    gr.Markdown('''
                    注意：最多支持生成6张图片!(You may generate a maximum of 6 photos at one time!)
                        ''')

        display_button = gr.Button('开始生成(Start!)')

        with gr.Box():
            infer_progress = gr.Textbox(label="生成进度(Progress)", value="当前无生成任务(No task)", interactive=False)
        with gr.Box():
            gr.Markdown('生成结果(Result)')
            output_images = gr.Gallery(label='Output', show_label=False).style(columns=3, rows=2, height=600,
                                                                               object_fit="contain")
                                                                               
        style_model.change(update_cloth, style_model, [cloth_style, pos_prompt])
        cloth_style.change(update_prompt, [style_model, cloth_style], [pos_prompt])
        display_button.click(fn=launch_pipeline,
                             inputs=[uuid, pos_prompt, user_models, num_images, style_model, multiplier_style],
                             outputs=[uuid, infer_progress, output_images])

    return demo


with gr.Blocks(css='style.css') as demo:
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
        with gr.TabItem('\N{rocket}形象定制(Train)'):
            train_input(uuid)
        with gr.TabItem('\N{party popper}形象体验(Inference)'):
            inference_input(uuid)

demo.queue(status_update_rate=1, api_open=False).launch(share=True, show_error=True)


