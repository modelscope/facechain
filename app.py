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
from modelscope import snapshot_download

from facechain.inference import GenPortrait
from facechain.inference_inpaint import GenPortraitInpaint
from facechain.data_process.preprocessing import get_popular_prompts
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, styles, cloth_prompt, \
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


def update_cloth(style_index):
    style = styles[style_index]
    prompts = []
    if style_index == 0:
        example_prompt = generate_pos_prompt(style['name'],
                                             cloth_prompt[0]['prompt'])
        multiplier_human = 0.95
        for prompt in cloth_prompt:
            prompts.append(prompt['name'])
    else:
        example_prompt = generate_pos_prompt(style['name'],
                                             style['add_prompt_style'])
        multiplier_human = style['multiplier_human']
        prompts.append(style['cloth_name'])
    return gr.Radio.update(choices=prompts,
                           value=prompts[0], visible=True), gr.Textbox.update(value=example_prompt), gr.Slider.update(value=multiplier_human)


def update_prompt(style_index, cloth_index):
    style = styles[style_index]
    if style_index == 0:
        pos_prompt = generate_pos_prompt(style['name'],
                                         cloth_prompt[cloth_index]['prompt'])
        multiplier_style = 0.25
    else:
        pos_prompt = generate_pos_prompt(style['name'],
                                         style['add_prompt_style'])
        multiplier_style = style['multiplier_style']
    return gr.Textbox.update(value=pos_prompt), gr.Slider.update(value=multiplier_style)

def update_pose_model(pose_image):
    if pose_image is None:
        return gr.Radio.update(value=pose_models[0]['name'])
    else:
        return gr.Radio.update(value=pose_models[1]['name'])

def update_optional_styles(base_model_index):
    style_list = base_models[base_model_index]['style_list']
    optional_styles = '\n'.join(style_list)
    return gr.Textbox.update(value=optional_styles)

def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def train_lora_fn(base_model_path=None, revision=None, sub_path=None, output_img_dir=None, work_dir=None, ensemble=True, enhance_lora=False, photo_num=0):
    validation_prompt, _ = get_popular_prompts(output_img_dir)
    torch.cuda.empty_cache()
    
    lora_r = 4 if not enhance_lora else 128
    lora_alpha = 32 if not enhance_lora else 64
    max_train_steps = min(photo_num * 200, 800)
    if ensemble:
        os.system(
            f'''
                PYTHONPATH=. accelerate launch facechain/train_text_to_image_lora.py \
                --pretrained_model_name_or_path="{base_model_path}" \
                --output_dataset_name="{output_img_dir}" \
                --caption_column="text" --resolution=512 \
                --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps={max_train_steps} --checkpointing_steps=100 \
                --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 --output_dir="{work_dir}" \
                --lora_r={lora_r} --lora_alpha={lora_alpha} \
                --validation_prompt="{validation_prompt}" \
                --validation_steps=100 \
                --template_dir="resources/inpaint_template" \
                --template_mask \
                --merge_best_lora_based_face_id \
                --revision="{revision}" \
                --sub_path="{sub_path}" \
            '''
        )
    else:
        os.system(
            f'PYTHONPATH=. accelerate launch facechain/train_text_to_image_lora.py --pretrained_model_name_or_path={base_model_path} '
            f'--revision={revision} --sub_path={sub_path} '
            f'--output_dataset_name={output_img_dir} --caption_column="text" --resolution=512 '
            f'--random_flip --train_batch_size=1 --num_train_epochs=200 --checkpointing_steps=5000 '
            f'--learning_rate=1.5e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 --seed=42 --output_dir={work_dir} '
            f'--lora_r={lora_r} --lora_alpha={lora_alpha} --lora_text_encoder_r=32 --lora_text_encoder_alpha=32 --resume_from_checkpoint="fromfacecommon"')

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
                    base_model_index=None,
                    user_model=None,
                    num_images=1,
                    style_model=None,
                    multiplier_style=0.25,
                    multiplier_human=0.85,
                    pose_model=None,
                    pose_image=None
                    ):
    # Check base model
    if base_model_index == None:
        raise gr.Error('请选择基模型(Please select the base model)！')
    
    # Check output model
    if not user_model:
        raise gr.Error('请选择产出模型(Please select the output model)！')
    
    # Check style model
    if style_model == None:
        raise gr.Error('请选择风格模型(Please select the style model)！')

    base_model = base_models[base_model_index]['model_id']
    revision = base_models[base_model_index]['revision']
    sub_path = base_models[base_model_index]['sub_path']
    
    before_queue_size = 0
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

    train_file = os.path.join(lora_model_path, 'pytorch_lora_weights.bin')
    if not os.path.exists(train_file):
        raise gr.Error('您还没有进行形象定制，请先进行训练。(Training is required before inference.)')


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
                yield [uuid, "排队等待资源中，前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5),
                        None]
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

        yield [uuid, "生成完毕(Generation done)！", outputs_RGB]
    else:
        yield [uuid, "生成失败，请重试(Generation failed, please retry)！", outputs_RGB]


def launch_pipeline_inpaint(uuid,
                          base_model_index=None,
                          user_model=None,
                          selected_template_images=None,
                          append_pos_prompt=None,
                          select_face_num=1,
                          first_control_weight=0.5,
                          second_control_weight=0.1,
                          final_fusion_ratio=0.5,
                          use_fusion_before=True,
                          use_fusion_after=True):
    before_queue_size = 0
    before_done_count = inference_done_count

    # Check base model
    if base_model_index == None:
        raise gr.Error('请选择基模型(Please select the base model)！')
    
    # Check output model
    if not user_model:
        raise gr.Error('请选择产出模型(Please select the output model)！')

    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            return "请登陆后使用! (Please login first)"
        else:
            uuid = 'qw'

    if isinstance(selected_template_images, str):
        if len(selected_template_images) == 0:
            raise gr.Error('请选择一张模板(Please select 1 template)')

    base_model = base_models[base_model_index]['model_id']
    revision = base_models[base_model_index]['revision']
    sub_path = base_models[base_model_index]['sub_path']
    output_model_name = 'personalization_lora'
    instance_data_dir = os.path.join('/tmp', uuid, 'training_data', base_model, user_model)

    # we use ensemble model, if not exists fallback to original lora
    lora_model_path = f'/tmp/{uuid}/{base_model}/{user_model}/ensemble/'
    if not os.path.exists(lora_model_path):
        lora_model_path = f'/tmp/{uuid}/{base_model}/{user_model}/'

    gen_portrait_inpaint = GenPortraitInpaint(crop_template=False, short_side_resize=512)
    
    cache_model_dir = snapshot_download("bubbliiiing/controlnet_helper", revision="v2.2")

    with ProcessPoolExecutor(max_workers=5) as executor:
        future = executor.submit(gen_portrait_inpaint, base_model, lora_model_path, instance_data_dir, \
                                 selected_template_images, cache_model_dir, select_face_num, first_control_weight, \
                                 second_control_weight, final_fusion_ratio, use_fusion_before, use_fusion_after, \
                                 sub_path=sub_path, revision=revision)

        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = inference_done_count
                to_wait = before_queue_size - (cur_done_count - before_done_count)
                yield ["排队等待资源中，前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5),
                       None]
            else:
                yield ["生成中, 请耐心等待(Generating)...", None]
            time.sleep(1)

    outputs = future.result()
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))
    image_path = './lora_result.png'
    if len(outputs) > 0:
        result = concatenate_images(outputs)
        cv2.imwrite(image_path, result)

        yield ["生成完毕(Generation done)！", outputs_RGB]
    else:
        yield ["生成失败，请重试(Generation failed, please retry)！", outputs_RGB]


class Trainer:
    def __init__(self):
        pass

    def run(
            self,
            uuid: str,
            ensemble: bool,
            enhance_lora: bool,
            instance_images: list,
            base_model_index: int,
            output_model_name: str,
    ) -> Tuple[str, str]:
        # Check Cuda
        if not torch.cuda.is_available():
            raise gr.Error('CUDA不可用(CUDA not available)')

        # Check Instance Valid
        if instance_images is None:
            raise gr.Error('您需要上传训练图片(Please upload photos)！')
        
        # Check output model name
        if not output_model_name:
            raise gr.Error('请指定产出模型的名称(Please specify the output model name)！')
        
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
            raise gr.Error("产出模型名称已存在。")

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
                      ensemble=ensemble,
                      enhance_lora=enhance_lora,
                      photo_num=len(instance_images))

        message = f'训练已经完成！请切换至 [形象体验] 标签体验模型效果(Training done, please switch to the inference tab to generate photos.)'
        print(message)
        return uuid, message


def flash_model_list(uuid, base_model_index):
    base_model_path = base_models[base_model_index]['model_id']
    style_list = base_models[base_model_index]['style_list']

    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            return "请登陆后使用! (Please login first)"
        else:
            uuid = 'qw'

    folder_path = f"/tmp/{uuid}/{base_model_path}"
    folder_list = []
    if not os.path.exists(folder_path):
        return gr.Radio.update(choices=[]),gr.Dropdown.update(choices=style_list)
    else:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)

    return gr.Radio.update(choices=folder_list), gr.Dropdown.update(choices=style_list, value=style_list[0], visible=True)

def update_output_model(uuid, base_model_index):

    # Check base model
    if base_model_index == None:
        raise gr.Error('请选择基模型(Please select the base model)！')

    base_model_path = base_models[base_model_index]['model_id']

    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            return "请登陆后使用! (Please login first)"
        else:
            uuid = 'qw'

    folder_path = f"/tmp/{uuid}/{base_model_path}"
    folder_list = []
    if not os.path.exists(folder_path):
        return gr.Radio.update(choices=[]),gr.Dropdown.update(choices=style_list)
    else:
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                if os.path.exists(file_lora_path):
                    folder_list.append(file)

    return gr.Radio.update(choices=folder_list)


def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths


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

                    base_model_index = gr.Radio(label="基模型选择(Base model list)", choices=base_model_list, type="index",
                                       value=base_model_list[0])
                    
                    optional_style = '\n'.join(base_models[0]['style_list'])
                    
                    optional_styles = gr.Textbox(label="该基模型支持的风格(Styles supported by this base model.)", lines=3,
                                        value=optional_style, interactive=False)
                    
                    output_model_name = gr.Textbox(label="产出模型名称(Output model name)", value='test', lines=1)
 
                    gr.Markdown('训练图片(Training photos)')
                    instance_images = gr.Gallery()
                    upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                    file_count="multiple")

                    clear_button = gr.Button("清空图片(Clear photos)")
                    clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                    upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images,
                                         queue=False)
                    
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

        with gr.Box():
            with gr.Row():
                ensemble = gr.Checkbox(label='人物LoRA融合（Ensemble）', value=False)
                enhance_lora = gr.Checkbox(label='LoRA增强（LoRA-Enhancement）', value=False)
            gr.Markdown(
                '''
                - 人物LoRA融合（Ensemble）：选择训练中几个最佳人物LoRA融合。提升相似度或在艺术照生成模式下建议勾选 - Allow fusion of multiple LoRAs during training. Recommended for enhanced-similarity or using with Inpaint mode.
                - LoRA增强（LoRA-Enhancement）：扩大LoRA规模，生成图片更贴近用户，至少5张以上多图训练或者艺术照生成模式建议勾选 - Boost scale of LoRA to enhance output resemblance with input. Recommended for training with more than 5 pics or using with Inpaint mode. 
                '''
            )

        run_button = gr.Button('开始训练（等待上传图片加载显示出来再点，否则会报错）'
                               'Start training (please wait until photo(s) fully uploaded, otherwise it may result in training failure)')

        with gr.Box():
            gr.Markdown('''
            请等待训练完成，请勿刷新或关闭页面。
            
            Please wait for the training to complete, do not refresh or close the page.
            ''')
            output_message = gr.Markdown()
        with gr.Box():
            gr.Markdown('''
            碰到抓狂的错误或者计算资源紧张的情况下，推荐直接在[NoteBook](https://modelscope.cn/my/mynotebook/preset)上进行体验。
            
            安装方法请参考：https://github.com/modelscope/facechain .
            
            If you are experiencing prolonged waiting time, you may try on [ModelScope NoteBook](https://modelscope.cn/my/mynotebook/preset) to prepare your dedicated environment.
                        
            You may refer to: https://github.com/modelscope/facechain for installation instruction.
            ''')
        base_model_index.change(fn=update_optional_styles,
                                inputs=[base_model_index],
                                outputs=[optional_styles],
                                queue=False)

        run_button.click(fn=trainer.run,
                         inputs=[
                             uuid,
                             ensemble,
                             enhance_lora,
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
                    with gr.Column(scale=3):
                        user_model = gr.Radio(label="产出模型(Output Model list)", choices=[], type="value")
                    with gr.Column(scale=1):
                        update_button = gr.Button('刷新产出模型(Update output model list)')

                style_model_list = []
                for style in styles:
                    style_model_list.append(style['name'])
                style_model = gr.Dropdown(choices=style_model_list, type="index", value=style_model_list[0], 
                                          label="风格模型(Style model)", visible=False)
                
                prompts = []
                for prompt in cloth_prompt:
                    prompts.append(prompt['name'])
                for style in styles[1:]:
                    prompts.append(style['cloth_name'])

                cloth_style = gr.Radio(choices=prompts, value=cloth_prompt[0]['name'],
                                       type="index", label="服装风格(Cloth style)", visible=False)
                pmodels = []
                for pmodel in pose_models:
                    pmodels.append(pmodel['name'])

                with gr.Accordion("高级选项(Advanced Options)", open=False):
                    pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3, 
                                            value=generate_pos_prompt(None, cloth_prompt[0]['prompt']),
                                            interactive=True)
                    multiplier_style = gr.Slider(minimum=0, maximum=1, value=0.25,
                                                 step=0.05, label='风格权重(Multiplier style)')
                    multiplier_human = gr.Slider(minimum=0, maximum=1.2, value=0.95,
                                                 step=0.05, label='形象权重(Multiplier human)')
                    pose_image = gr.Image(source='upload', type='filepath', label='姿态图片(Pose image)')
                    gr.Examples(pose_examples['man'], inputs=[pose_image], label='男性姿态示例')
                    gr.Examples(pose_examples['woman'], inputs=[pose_image], label='女性姿态示例')
                    pose_model = gr.Radio(choices=pmodels, value=pose_models[0]['name'],
                                          type="index", label="姿态控制模型(Pose control model)")
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
                                                                               
        style_model.change(update_cloth, style_model, [cloth_style, pos_prompt, multiplier_human], queue=False)
        cloth_style.change(update_prompt, [style_model, cloth_style], [pos_prompt, multiplier_style], queue=False)
        pose_image.change(update_pose_model, pose_image, [pose_model])
        base_model_index.change(fn=flash_model_list,
                                inputs=[uuid, base_model_index],
                                outputs=[user_model, style_model],
                                queue=False)
        update_button.click(fn=update_output_model,
                      inputs=[uuid, base_model_index],
                      outputs=[user_model],
                      queue=False)
        display_button.click(fn=launch_pipeline,
                             inputs=[uuid, pos_prompt, base_model_index, user_model, num_images, style_model, multiplier_style, multiplier_human,
                                     pose_model, pose_image],
                             outputs=[uuid, infer_progress, output_images])

    return demo


def inference_inpaint():
    """
        Inpaint Tab with Ensemble-Lora + MultiControlnet, support preset_template
        #TODO: Support user upload template && template check logits
    """
    preset_template = glob(os.path.join('resources/inpaint_template/*.jpg'))
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        # Initialize the GUI
        
        with gr.Row():
            with gr.Column():
                base_model_list = []
                for base_model in base_models:
                    base_model_list.append(base_model['name'])

                base_model_index = gr.Radio(
                    label="基模型选择(Base model list)",
                    choices=base_model_list,
                    type="index"
                )

                user_model = gr.Radio(label="产出模型(Output Model list)", choices=[], type="value")
                
                template_gallery_list = [(i, f"模板{idx + 1}") for idx, i in enumerate(preset_template)]
                gallery = gr.Gallery(template_gallery_list).style(grid=4, height=300)

                # new inplementation with gr.select callback function, only pick 1image at once
                def select_function(evt: gr.SelectData):
                    return [preset_template[evt.index]]

                selected_template_images = gr.Text(show_label=False, placeholder="Selected")
                gallery.select(select_function, None, selected_template_images)
                
                with gr.Accordion("高级选项(Advanced Options)", open=False):
                    append_pos_prompt = gr.Textbox(
                        label="提示语(Prompt)",
                        lines=3,
                        value='masterpiece, smile, beauty',
                        interactive=True
                    )
                    first_control_weight = gr.Slider(
                        minimum=0.35, maximum=0.6, value=0.45,
                        step=0.02, label='初始权重(Initial Control Weight)'
                    )

                    second_control_weight = gr.Slider(
                        minimum=0.04, maximum=0.2, value=0.1,
                        step=0.02, label='二次权重(Secondary Control Weight)'
                    )
                    final_fusion_ratio = gr.Slider(
                        minimum=0.2, maximum=0.8, value=0.5,
                        step=0.1, label='融合系数(Final Fusion Ratio)'
                    )
                    select_face_num = gr.Slider(
                        minimum=1, maximum=4, value=1,
                        step=1, label='生成数目(Number of Reference Faces)'
                    )
                    use_fusion_before = gr.Radio(
                        label="前融合(Apply Fusion Before)", type="value", choices=[True, False],
                        value=True
                    )
                    use_fusion_after = gr.Radio(
                        label="后融合(Apply Fusion After)", type="value", choices=[True, False],
                        value=True
                    )
                
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

        base_model_index.change(fn=update_output_model,
                                inputs=[uuid, base_model_index],
                                outputs=[user_model],
                                queue=False)
                        
        display_button.click(
            fn=launch_pipeline_inpaint,
            inputs=[uuid, base_model_index, user_model, selected_template_images, append_pos_prompt, select_face_num, first_control_weight,
                    second_control_weight,
                    final_fusion_ratio, use_fusion_before, use_fusion_after],
            outputs=[infer_progress, output_images]
        )
        
    return demo


with gr.Blocks(css='style.css') as demo:
	gr.Markdown("# <center> \N{fire} FaceChain Potrait Generation ([Github star it here](https://github.com/modelscope/facechain/tree/main) \N{whale}, [Paper cite it here](https://arxiv.org/abs/2308.14256) \N{whale})</center>")
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
        with gr.TabItem('\N{party popper}艺术照(Inpaint)'):
            inference_inpaint(uuid)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    demo.queue(status_update_rate=1).launch(share=True)

