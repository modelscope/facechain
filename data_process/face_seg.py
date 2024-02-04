from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
import numpy as np
import cv2
import sys
import os
from tqdm import tqdm 
filein_name = sys.argv[1]
filein = open(filein_name, 'r')
segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
save_root = './pure_face_mask'

if not os.path.exists(save_root):
    os.makedirs(save_root)

def get_mask_head(result):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    img_shape = masks[0].shape
    mask_hair = np.zeros(img_shape)
    mask_face = np.zeros(img_shape)
    mask_human = np.zeros(img_shape)
    for i in range(len(labels)):
        if scores[i] > 0.8:
            if labels[i] == 'Face':
                if np.sum(masks[i]) > np.sum(mask_face):
                    mask_face = masks[i]
            elif labels[i] == 'Human':
                if np.sum(masks[i]) > np.sum(mask_human):
                    mask_human = masks[i]
            elif labels[i] == 'Hair':
                if np.sum(masks[i]) > np.sum(mask_hair):
                    mask_hair = masks[i]
    mask_head = np.clip(mask_hair + mask_face, 0, 1)
    ksize = max(int(np.sqrt(np.sum(mask_face)) / 20), 1)
    kernel = np.ones((ksize, ksize))
    mask_head = cv2.dilate(mask_head, kernel, iterations=1) * mask_human
    _, mask_head = cv2.threshold((mask_head * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask_head, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    mask_head = np.zeros(img_shape).astype(np.uint8)
    cv2.fillPoly(mask_head, [contours[max_idx]], 255)
    mask_head = mask_head.astype(np.float32) / 255 
    mask_head = np.clip(mask_head + mask_face, 0, 1)
    mask_head = np.expand_dims(mask_head, 2) * 255
    return mask_head #[512, 512, 1] (0, 255)

for line in filein:
    img_path = line.strip().split()[0]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_CUBIC)
    filename = img_path.split('/')[-1]
    if img is None:
        print('imgNone: ', img_path)
        continue
    try:
        result = segmentation_pipeline(img_path)
        mask_head = get_mask_head(result) #[512, 512, 1] (0,1)
        cv2.imwrite(os.path.join(save_root, filename), mask_head)
        print(os.path.join(save_root, filename), 'has been seged!\n')
    except Exception as e:
        print(img_path, 'seg failed')
        continue

    

