# Copyright (c) Alibaba, Inc. and its affiliates.
import enum
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
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
    pose_models, pose_examples

training_done_count = 0
inference_done_count = 0

HOT_MODELS = [
    "\N{fire}数字身份(Digital Identity)",
]

FACE_TAGS = {'default'}
default_personalizaition_lora_name = "personalizaition_lora"


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
    return gr.Radio.update(choices=prompts,
                           value=prompts[0], visible=True), gr.Textbox.update(value=example_prompt)


def update_prompt(style_index, cloth_index):
    if style_index == 0:
        pos_prompt = generate_pos_prompt(styles[style_index]['name'],
                                         cloth_prompt[cloth_index]['prompt'])
    else:
        pos_prompt = generate_pos_prompt(styles[style_index]['name'],
                                         styles[style_index]['add_prompt_style'])
    return gr.Textbox.update(value=pos_prompt)


def update_pose_model(pose_image):
    if pose_image is None:
        return gr.Radio.update(value=pose_models[0]['name'])
    else:
        return gr.Radio.update(value=pose_models[1]['name'])


def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def train_lora_fn(foundation_model_path=None, revision=None, output_img_dir=None, work_dir=None, ensemble=True,
                  enhance_lora=False, photo_num=0):
    validation_prompt, _ = get_popular_prompts(output_img_dir)
    torch.cuda.empty_cache()

    lora_r = 4 if not enhance_lora else 128
    lora_alpha = 32 if not enhance_lora else 64
    max_train_steps = min(photo_num * 200, 800)
    if ensemble:
        os.system(
            f'''
                PYTHONPATH=. accelerate launch facechain/train_text_to_image_lora.py \
                --pretrained_model_name_or_path="{foundation_model_path}" \
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
                --sub_path="film/film" \
            '''
        )
    else:
        os.system(
            f'PYTHONPATH=. accelerate launch facechain/train_text_to_image_lora.py --pretrained_model_name_or_path={foundation_model_path} '
            f'--revision={revision} --sub_path="film/film" '
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
                    user_models,
                    num_images=1,
                    style_model=None,
                    multiplier_style=0.25,
                    pose_model=None,
                    pose_image=None,
                    face_tag = None
                    ):
    base_model = 'ly261666/cv_portrait_model'
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

    print("-------user_models: ", user_models)
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            return "请登陆后使用! (Please login first)"
        else:
            uuid = 'qw'

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False

    output_model_name = default_personalizaition_lora_name
    if face_tag is not None:
        output_model_name = output_model_name + "_" + face_tag.replace(" ", "_")

    instance_data_dir = os.path.join('/tmp', uuid, 'training_data', output_model_name)
    lora_model_path = f'/tmp/{uuid}/{output_model_name}/ensemble'
    if not os.path.exists(lora_model_path):
        lora_model_path = f'/tmp/{uuid}/{output_model_name}/'

    train_file = os.path.join(lora_model_path, 'pytorch_lora_weights.bin')
    if not os.path.exists(train_file):
        raise gr.Error('您还没有进行形象定制，请先进行训练。(Training is required before inference.)')

    gen_portrait = GenPortrait(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path,
                               multiplier_style, use_main_model,
                               use_face_swap, use_post_process,
                               use_stylization)

    num_images = min(6, num_images)

    with ProcessPoolExecutor(max_workers=5) as executor:
        future = executor.submit(gen_portrait, instance_data_dir,
                                 num_images, base_model, lora_model_path, 'film/film', 'v2.0')
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


def launch_pipeline_inpaint(uuid,
                            selected_template_images,
                            append_pos_prompt,
                            select_face_num=1,
                            first_control_weight=0.5,
                            second_control_weight=0.1,
                            final_fusion_ratio=0.5,
                            use_fusion_before=True,
                            use_fusion_after=True,
                            face_tag=None):
    before_queue_size = 0
    before_done_count = inference_done_count

    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            return "请登陆后使用! (Please login first)"
        else:
            uuid = 'qw'

    if isinstance(selected_template_images, str):
        if len(selected_template_images) == 0:
            raise gr.Error('请选择一张模板(Please select 1 template)')

    base_model = 'ly261666/cv_portrait_model'
    output_model_name = default_personalizaition_lora_name
    if face_tag is not None:
        output_model_name = output_model_name + "_" + face_tag.replace(" ", "_")

    instance_data_dir = os.path.join('/tmp', uuid, 'training_data', output_model_name)

    # we use ensemble model, if not exists fallback to original lora
    lora_model_path = f'/tmp/{uuid}/{output_model_name}/ensemble/'
    if not os.path.exists(lora_model_path):
        lora_model_path = f'/tmp/{uuid}/{output_model_name}/'

    gen_portrait_inpaint = GenPortraitInpaint(crop_template=False, short_side_resize=512)

    cache_model_dir = snapshot_download("bubbliiiing/controlnet_helper", revision="v2.2")

    with ProcessPoolExecutor(max_workers=5) as executor:
        future = executor.submit(gen_portrait_inpaint, base_model, lora_model_path, instance_data_dir, \
                                 selected_template_images, cache_model_dir, select_face_num, first_control_weight, \
                                 second_control_weight, final_fusion_ratio, use_fusion_before, use_fusion_after,
                                 sub_path='film/film', revision='v2.0')
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
            face_tag
    ) -> str:
        # Check Cuda
        if not torch.cuda.is_available():
            raise gr.Error('CUDA不可用(CUDA not available)')

        # Check Instance Valid
        if instance_images is None:
            raise gr.Error('您需要上传训练图片(Please upload photos)！')

        # Limit input Image
        if len(instance_images) > 20:
            raise gr.Error('请最多上传20张训练图片(20 images at most!)')

        # Check UUID & Studio
        if not uuid:
            if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
                return "请登陆后使用(Please login first)! "
            else:
                uuid = 'qw'

        output_model_name = default_personalizaition_lora_name
        if face_tag is not None:
            output_model_name = output_model_name + "_" + face_tag.replace(" ", "_")
            FACE_TAGS.add(face_tag)

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
        print("instance_data_dir", instance_data_dir)
        train_lora_fn(foundation_model_path='ly261666/cv_portrait_model',
                      revision='v2.0',
                      output_img_dir=instance_data_dir,
                      work_dir=work_dir,
                      ensemble=ensemble,
                      enhance_lora=enhance_lora,
                      photo_num=len(instance_images))

        message = f'训练已经完成！请切换至 [形象体验] 标签体验模型效果(Training done, please switch to the inference tab to generate photos.)'
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


def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths


def train_input():
    trainer = Trainer()

    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('训练图片(Training photos)')
                    instance_images = gr.Gallery()
                    with gr.Row():
                        upload_button = gr.UploadButton("选择图片上传(Upload photos)", file_types=["image"],
                                                        file_count="multiple")

                        clear_button = gr.Button("清空图片(Clear photos)")
                    clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                    upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images,
                                         queue=False)

                    gr.Markdown('''
                        - Step 1. 上传计划训练的图片，3~10张头肩照（注意：请避免图片中出现多人脸、脸部遮挡等情况，否则可能导致效果异常）
                        - Step 2. 给人脸打标签（可选）
                        - Step 3. 点击 [开始训练] ，启动形象定制化训练，约需15分钟，请耐心等待～
                        - Step 4. 切换至 [形象体验] ，生成你的风格照片
                        ''')
                    gr.Markdown('''
                        - Step 1. Upload 3-10 headshot photos of yours (Note: avoid photos with multiple faces or face obstruction, which may lead to non-ideal result).
                        - Step 2. Add tag for current face(optional).
                        - Step 3. Click [Train] to start training for customizing your Digital-Twin, this may take up-to 15 mins.
                        - Step 4. Switch to [Inference] Tab to generate stylized photos.
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
        with gr.Row():
            face_tag = gr.Text(label='人脸标签(Face tag)', value='default', interactive=True)
            gr.Markdown('''可以给当前人脸设置标签
                           Tag can be set for current face
                        ''')

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

        run_button.click(fn=trainer.run,
                         inputs=[
                             uuid,
                             ensemble,
                             enhance_lora,
                             instance_images,
                             face_tag
                         ],
                         outputs=[output_message])

    return demo


def inference_input():
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        with gr.Row():
            with gr.Column():
                user_models = gr.Radio(label="模型选择(Model list)", choices=HOT_MODELS, type="value",
                                       value=HOT_MODELS[0])
                with gr.Row():
                    face_tag = gr.Dropdown(choices=FACE_TAGS, value='default', type="value", label="人脸标签选择(Face tag)",
                                           interactive=True)
                    update_button = gr.Button('刷新人脸(Refresh face tag)')
                    update_button.click(dropdown_face_tag_list, inputs=[], outputs=[face_tag])

                style_model_list = []
                for style in styles:
                    style_model_list.append(style['name'])
                style_model = gr.Dropdown(choices=style_model_list, type="index", label="风格模型(Style model)")

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
                    pos_prompt = gr.Textbox(label="提示语(Prompt)", lines=3, interactive=True)
                    multiplier_style = gr.Slider(minimum=0, maximum=1, value=0.25,
                                                 step=0.05, label='风格权重(Multiplier style)')
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

        style_model.change(update_cloth, style_model, [cloth_style, pos_prompt])
        cloth_style.change(update_prompt, [style_model, cloth_style], [pos_prompt])
        pose_image.change(update_pose_model, pose_image, [pose_model])
        display_button.click(fn=launch_pipeline,
                             inputs=[uuid, pos_prompt, user_models, num_images, style_model, multiplier_style,
                                     pose_model, pose_image, face_tag],
                             outputs=[infer_progress, output_images])

    return demo

def dropdown_face_tag_list():
    return gr.Dropdown.update(choices=FACE_TAGS)

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
                user_models = gr.Radio(
                    label="Model Selection",
                    choices=HOT_MODELS,
                    type="value",
                    value=HOT_MODELS[0]
                )

                # 人脸
                with gr.Row():
                    face_tag = gr.Dropdown(choices=FACE_TAGS, value='default', type="value", label="人脸标签选择(Face tag)",
                                           interactive=True)
                    update_button = gr.Button('刷新人脸(Refresh face tag)')
                    update_button.click(dropdown_face_tag_list, inputs=[], outputs=[face_tag])

                template_gallery_list = [(i, f"模板{idx + 1}") for idx, i in enumerate(preset_template)]
                gallery = gr.Gallery(template_gallery_list).style(grid=4, height=300)

                # new inplementation with gr.select callback function, only pick 1image at once
                def select_function(evt: gr.SelectData):
                    return [preset_template[evt.index]]

                selected_template_images = gr.Text(show_label=False, placeholder="Selected")
                gallery.select(select_function, None, selected_template_images)

                with gr.Accordion("Advanced Options", open=False):
                    append_pos_prompt = gr.Textbox(
                        label="Prompt",
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

        display_button = gr.Button('Start Generation')
        with gr.Box():
            infer_progress = gr.Textbox(
                label="生成(Generation Progress)",
                value="No task currently",
                interactive=False
            )
        with gr.Box():
            gr.Markdown('Generated Results')
            output_images = gr.Gallery(
                label='输出(Output)',
                show_label=False
            ).style(columns=3, rows=2, height=600, object_fit="contain")

        display_button.click(
            fn=launch_pipeline_inpaint,
            inputs=[uuid, selected_template_images, append_pos_prompt, select_face_num, first_control_weight,
                    second_control_weight,
                    final_fusion_ratio, use_fusion_before, use_fusion_after, face_tag],
            outputs=[infer_progress, output_images]
        )

    return demo


with gr.Blocks(css='style.css') as demo:
    with gr.Tabs():
        with gr.TabItem('\N{rocket}形象定制(Train)'):
            train_input()
        with gr.TabItem('\N{party popper}形象体验(Inference)'):
            inference_input()
        with gr.TabItem('\N{party popper}艺术照(Inpaint)'):
            inference_inpaint()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    demo.queue(status_update_rate=1).launch(share=True)
