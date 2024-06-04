import cv2
import os
import json
from facechain.inference_inpaint_fact import GenPortrait_inpaint

num_faces = 1
selected_face = 1
strength = 0.6
inpaint_img = 'poses/man/pose1.png'
input_img_path = 'poses/man/pose2.png'
num_generate = 1
output_dir = './generated_inpaint'

pos_prompt = 'raw photo, masterpiece, simple background, solo, medium shot, high detail face, photorealistic, best quality, wearing T-shirt'
neg_prompt = 'nsfw, paintings, sketches, (worst quality:2), (low quality:2) ' \
                'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character'
output_img_size = 512

gen_portrait = GenPortrait_inpaint()

outputs = gen_portrait(inpaint_img, strength, output_img_size, num_faces, selected_face, pos_prompt, neg_prompt, input_img_path, num_generate)
os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)
