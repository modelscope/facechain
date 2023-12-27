import cv2
import os
import json
from facechain.utils import snapshot_download
from facechain.inference_inpaint import GenPortrait_inpaint
from facechain.constants import base_models

num_faces = 1
multiplier_style = 0.05
multiplier_human = 0.95
strength = 0.65
output_img_size = 512

model_dir = snapshot_download('ly261666/cv_wanx_style_model', revision='v1.0.3')
style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_frommajicreal.safetensors')

pos_prompt = 'raw photo, masterpiece, chinese, simple background, high-class pure color background, solo, medium shot, high detail face, photorealistic, best quality, wearing T-shirt'
neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
             'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'

instance_data_dir_A = './processed_anze'
lora_model_path_A = "./output_anze"
instance_data_dir_B = None
lora_model_path_B = None

in_path = "./resources/inpaint_template/4.jpg"
output_dir = './inpaint_anze_rst'
base_model = base_models[0]
use_main_model = True
use_face_swap = True
use_post_process = True
use_stylization = False

gen_portrait = GenPortrait_inpaint(in_path, strength, num_faces,
                                pos_prompt, neg_prompt, style_model_path,
                                multiplier_style, multiplier_human, use_main_model,
                                use_face_swap, use_post_process,
                                use_stylization)

outputs = gen_portrait(instance_data_dir_A, instance_data_dir_B, base_model['model_id'],\
                             lora_model_path_A, lora_model_path_B, sub_path=base_model['sub_path'], revision= base_model['revision'])

os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)
