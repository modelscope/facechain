from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import os
import cv2
from PIL import Image
import sys

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

lstin = open(sys.argv[1], 'r')
crop_dir = sys.argv[2]
caption_dir = sys.argv[3]
for line in lstin:
    img_path = line.strip().split(' ')[0]
    img_path = os.path.join(crop_dir, img_path.split('/')[-1].split('.')[0]+'.jpg')
    img = Image.open(img_path)
    try:
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(img_path, generated_text)
        caption_out = open(os.path.join(caption_dir, img_path.split('/')[-1].rsplit('.', 1)[0]+'.txt'), 'w')
        caption_out.write(generated_text+'\n')

    except Exception as e:
        print(img_path, 'no caption result')
        continue
