import os
import cv2
import sys
import modelscope

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def get_center_dis(rect, w, h):
    cx = w / 2
    cy = h / 2
    crx = (rect[0]+rect[2])/2
    cry = (rect[1]+rect[3])/2
    dis = (cx-crx)*(cx-crx) + (cy-cry)*(cy-cry)
    return dis

def get_rect_area(rect):
    rw = rect[2]-rect[0]
    rh = rect[3]-rect[1]
    return rw * rh

def to_rgb(img):
    h, w = img.shape
    ret = np.empty((h, w, 3), dtype = np.uint8)
    ret[:,:,0] = ret[:,:,1] = ret[:,:,2] = img
    return ret

retina_face_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')

filein_name = sys.argv[1]
fileout_name = sys.argv[2]
multiout_name = sys.argv[3]

filein = open(filein_name, 'r')
fileout = open(fileout_name, 'w')
multiout = open(multiout_name, 'w')
for line in filein:
    img_path = line.strip().split()[0]
    img = cv2.imread(img_path)
    if img is None:
        print('imgNone::', img_path)
        continue
    if img.ndim == 2:
        img = to_rgb(img)
    h,w,c = img.shape
    scale = 1
    if h > 500 or w > 500:
        if h > w:
            scale = h / 500.
            newh = 500
            neww = int(w / scale)
        else:
            scale = w / 500.
            neww = 500
            newh = int(h / scale)
        img = cv2.resize(img, (neww, newh))
        # print('resize::', img_path)
    result = retina_face_detection(img)
    #print(f'face detection output: {result}.')
    scores = result['scores']
    rects = result['boxes']
    keypoints = result['keypoints']
    if len(scores) < 1:
        continue
    
    if len(scores) > 1:
        outstr = '%s'%(img_path)
        for i in range(len(scores)):
            for rv in rects[i]:
                outstr += ' %d'%(int(rv*scale))
            for pv in keypoints[i]:
                outstr += ' %f'%(pv*scale)
            outstr += ' %f'%scores[i]
        outstr += '\n'
        multiout.write(outstr)
        continue

    maxarea = 0
    maxidx = 0
    for i in range(len(scores)):
        score = scores[i]
        rect = rects[i]
        point = keypoints[i]
        area = get_rect_area(rect)
        maxarea, maxidx = (maxarea, maxidx) if area < maxarea else (area, i)
    selected_score = scores[maxidx]
    selected_rect = rects[maxidx]
    selected_point = keypoints[maxidx]

    outstr = '%s'%(img_path)
    for rv in selected_rect:
        outstr += ' %d'%(int(rv*scale))
    for pv in selected_point:
        outstr += ' %f'%(pv*scale)
    outstr += ' %f\n'%selected_score
    fileout.write(outstr)

