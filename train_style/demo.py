import gradio as gr
import os
import json
import shutil
from PIL import Image
# import sys
# # 当前文件目录加入路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .deepbooru import DeepDanbooru
from .convert_lora import convert_lora
from facechain.utils import project_dir, set_spawn_method

def set_img(files, uuid, output_model_name):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    print("uuid: ", uuid)
    
    if not output_model_name:
        raise gr.Error("请指定风格lora的名称(Please specify the style LoRA name)！")
    temp_img_dir = os.path.join(project_dir, 'workspace', uuid, 'train_imgs', output_model_name)
    if os.path.exists(temp_img_dir):
        raise gr.Error("风格lora名称已存在(This style lora name already exists.)")
    
    os.makedirs(temp_img_dir, exist_ok=True)
    train_idx = 0
    train_folder = temp_img_dir + f'/{train_idx:06d}'
    shutil.rmtree(train_folder, ignore_errors=True)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder, exist_ok=True)

    new_imgs = []
    for file in files:
        file_path = file.orig_name
        file_name = os.path.basename(file_path)
        prompt = ''
        new_path = os.path.join(train_folder, file_name)
        # 放到临时目录
        if os.path.exists(new_path):
            os.remove(new_path)
        os.system('cp {} {}'.format(file_path, new_path))
        # 居中裁剪图片到512*512
        cut_img(new_path)
        new_imgs.append([new_path, prompt])
    
    return train_folder, gr.Gallery.update(value=new_imgs, visible=True), gr.Button.update(interactive=False)

def init_tag():
    model = DeepDanbooru()
    model.start()
    return model

def cut_img(img_path):
    '''
    居中裁剪图片到512*512
    '''
    import cv2
    # 按短边等比例缩放
    def resize_img(img, size=512):
        h, w = img.shape[:2]
        if h < w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)
        return cv2.resize(img, (new_w, new_h))
    # 居中裁剪
    def crop_img(img, size=512):
        h, w = img.shape[:2]
        x = (w - size) // 2
        y = (h - size) // 2
        return img[y:y+size, x:x+size]
    # 读取图片
    img = cv2.imread(img_path)
    # 缩放图片
    img = resize_img(img)
    # 裁剪图片
    img = crop_img(img)
    cv2.imwrite(img_path, img)
    return img

def train_lora(uuid, output_model_name, prompt_input, train_folder, gallery, rank, num_train_epochs):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
    print("uuid: ", uuid)
    print(prompt_input, train_folder, gallery, rank, num_train_epochs)
    out_jsonl = ''
    num_train_img = len(gallery)
    tags_all = []
    add_prompt_style = []
    for item in gallery:
        if len(item) == 2:
            file = item[0]
            prompt = item[1]
        else:
            file = item
        if 'name' in file:
            f_path = file['name']
            f_name = os.path.basename(f_path)
            text = prompt_input
            if prompt:
                text = prompt_input + ', ' + prompt
            j = {
                "file_name": f_name,
                "text": text
            }
            tags_all.extend(text.split(', '))
        out_jsonl += json.dumps(j, ensure_ascii=False) + '\n'

    with open(f'{train_folder}/metadata.jsonl', 'w') as f:
        f.write(out_jsonl)
    
    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * num_train_img:
            if not tag in ['upper_body', 'raw photo', 'masterpiece', 'solo', 'medium shot', 'high detail face', 'photorealistic', 'best quality', '1girl']:
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)
    
    prompt_text = add_prompt_style[0]
    for i in range(1, len(add_prompt_style)):
        prompt_text = prompt_text + ', ' + add_prompt_style[i]
    
    set_spawn_method()
    
    command = [
        'python', f'{project_dir}/train_style/train_text_to_image_lora.py',
        f'--pretrained_model_name_or_path=ly261666/cv_portrait_model',
        f'--dataset_name={train_folder}',
        f'--caption_column=text',
        f'--resolution=512',
        f'--random_flip',
        f'--train_batch_size=1',
        f'--num_train_epochs={int(num_train_epochs)}',
        f'--checkpointing_steps=5000',
        f'--learning_rate=1e-04',
        f'--lr_scheduler=constant',
        f'--lr_warmup_steps=0',
        f'--seed=42',
        f'--output_dir={project_dir}/workspace/{uuid}/style_lora/{output_model_name}',
        f'--report_to=wandb',
        f'--rank={int(rank)}'
    ]
    
    import subprocess
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the command: {e}")
        raise gr.Error("训练失败（Training failed）")
    
    convert_lora(f'{project_dir}/workspace/{uuid}/style_lora/{output_model_name}/pytorch_lora_weights.safetensors', f'{project_dir}/workspace/{uuid}/style_lora/{output_model_name}/lora_weights.safetensors')
    
    out_path = f'{project_dir}/workspace/{uuid}/style_lora/{output_model_name}/lora_weights.safetensors'
    out_txt_path = f'{project_dir}/workspace/{uuid}/style_lora/{output_model_name}/add_prompt_style.txt'
    f = open(out_txt_path, 'w')
    f.write(prompt_text)
    f.close()
    
    return gr.Files.update(value=[out_path], visible=True), gr.Text.update(value=prompt_text, visible=True)

def set_prompt():
    return gr.Button.update(interactive=True)

