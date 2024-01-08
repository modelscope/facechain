import cv2
import os
import json
from facechain.utils import snapshot_download
from facechain.inference_inpaint import GenPortrait_inpaint
from facechain.constants import base_models
from facechain.inference_tryon import GenPortrait_tryon

base_model_index = 0
base_model = base_models[base_model_index]['model_id']
revision = base_models[base_model_index]['revision']
sub_path = base_models[base_model_index]['sub_path']
multiplier_style = 0.05
multiplier_human = 0.95
character_model = 'ly261666/cv_portrait_model'
do_inpaint = 0

background_prompt = 'on the beach, near the sea'
# 'on the beach, near the sea',
# 'in the bar, restaurant',
# 'city background, street',
# 'in the woods'

tmp_character_model = base_models[base_model_index]['model_id']
if tmp_character_model != character_model:
    tmp_character_model = 'ly261666/cv_portrait_model'

model_dir = snapshot_download('ly261666/cv_wanx_style_model', revision='v1.0.3')
style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_frommajicreal.safetensors')

pos_prompt = 'raw photo, masterpiece, ' + background_prompt + ', (completely nude:2), solo, medium shot, high detail face, photorealistic, best quality'
neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
             'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'

instance_data_dir = './processed_anze'
lora_model_path ="./output_anze"
in_path = "./resources/tryon_garment/garment4.png"
output_dir_tryon = './tryon_anze_rst'
output_dir_inpaint = './inpaint_anze_rst'

use_main_model = True
use_face_swap = True
use_post_process = True
use_stylization = False

gen_portrait = GenPortrait_tryon(in_path, 1.0,
                                 pos_prompt, neg_prompt, style_model_path,
                                 multiplier_style, multiplier_human, use_main_model,
                                 use_face_swap, use_post_process,
                                 use_stylization)

future = gen_portrait(instance_data_dir, base_model, \
                             lora_model_path, sub_path=sub_path, revision=revision)

os.makedirs(output_dir_tryon, exist_ok=True)
for i, out_tmp in enumerate(future):
    cv2.imwrite(os.path.join(output_dir_tryon, f'{i}.png'), out_tmp)

print("done.  tryon")

outputs = future
if do_inpaint == 0:
    cv2.imwrite('tmp_tryon.png', outputs[0])
    pos_prompt = 'raw photo, masterpiece, chinese, simple background, high-class pure color background, solo, medium shot, high detail face, photorealistic, best quality, wearing T-shirt'
    neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
                 'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'
    gen_portrait = GenPortrait_inpaint('tmp_tryon.png', 0.65, 1,
                                       pos_prompt, neg_prompt, style_model_path,
                                       multiplier_style, multiplier_human, use_main_model,
                                       use_face_swap, use_post_process,
                                       use_stylization)
    outputs = gen_portrait(instance_data_dir, None, base_model, \
                                 lora_model_path, None, sub_path=sub_path, revision=revision)

    os.makedirs(output_dir_inpaint, exist_ok=True)
    for i, out_tmp in enumerate(outputs):
        cv2.imwrite(os.path.join(output_dir_inpaint, f'{i}.png'), out_tmp)
print("done.  tryon inpaint")