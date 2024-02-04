from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os 
import cv2
import numpy as np
import sys

def to_rgb(img):
    h, w = img.shape
    ret = np.empty((h, w, 3), dtype = np.uint8)
    ret[:,:,0] = ret[:,:,1] = ret[:,:,2] = img
    return ret

face_human_hand_detection = pipeline(Tasks.face_human_hand_detection, model='damo/cv_nanodet_face-human-hand-detection')

filein_name = sys.argv[1]
fileout_name = sys.argv[2]
multiout_name = sys.argv[3]

filein = open(filein_name, 'r')
fileout = open(fileout_name, 'w')
multiout = open(multiout_name, 'w')
for line in filein:
    img_path = line.strip().split()[0]
    filename = img_path.split('/')[-1]
    img = cv2.imread(img_path)
    if img is None:
        print('imgNone: ', img_path)
        continue
    if img.ndim == 2:
        img = to_rgb(img)

    result_status = face_human_hand_detection({'input_path': img_path})
    labels = result_status[OutputKeys.LABELS]
    boxes = result_status[OutputKeys.BOXES]
    scores = result_status[OutputKeys.SCORES]
    
    # if labels.count(1) == 1 and labels.count(2) == 0:
    if labels.count(1) == 1:
        txt_input = filename + '\n'
        fileout.write(txt_input)
        print('Detect one Face' + txt_input)
    else:
        txt_input = filename + '\n'
        multiout.write(txt_input)
        print('Detect more than one Face' + txt_input)
    
