# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import math
import os
import shutil

import cv2
import numpy as np
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from tqdm import tqdm

from .deepbooru import DeepDanbooru
from .face_process_utils import call_face_crop


def get_popular_prompts(train_img_dir):
    """ check original Training datasets to calculate popular prompts.
    """
    train_dir = str(train_img_dir) + '_labeled'
    neg_prompt = ''
    add_prompt_style = []
    f = open(os.path.join(train_dir, 'metadata.jsonl'), 'r')
    tags_all = []
    cnt = 0
    cnts_trigger = np.zeros(6)
    for line in f:
        cnt += 1
        data = json.loads(line)['text'].split(', ')
        tags_all.extend(data)
        if data[1] == 'a boy':
            cnts_trigger[0] += 1
        elif data[1] == 'a girl':
            cnts_trigger[1] += 1
        elif data[1] == 'a handsome man':
            cnts_trigger[2] += 1
        elif data[1] == 'a beautiful woman':
            cnts_trigger[3] += 1
        elif data[1] == 'a mature man':
            cnts_trigger[4] += 1
        elif data[1] == 'a mature woman':
            cnts_trigger[5] += 1
        else:
            print('Error.')
    f.close()

    attr_idx = np.argmax(cnts_trigger)
    trigger_styles = ['a boy, children, ', 'a girl, children, ', 'a handsome man, ', 'a beautiful woman, ',
                    'a mature man, ', 'a mature woman, ']
    trigger_style = '<fcsks>, ' + trigger_styles[attr_idx]

    if attr_idx == 2 or attr_idx == 4:
        neg_prompt += ', children'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('hair' in tag or 'face' in tag or 'mouth' in tag or 'skin' in tag or 'smile' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)

    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''
    validation_prompt = trigger_style + add_prompt_style
    return validation_prompt, neg_prompt


def crop_and_resize(im, bbox):
    h, w, _ = im.shape
    thre = 0.35/1.15
    maxf = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    cx = (bbox[2] + bbox[0]) / 2
    cy = (bbox[3] + bbox[1]) / 2
    lenp = int(maxf / thre)
    yc = 0.5/1.15
    xc = 0.5
    xmin = int(cx - xc * lenp)
    xmax = xmin + lenp
    ymin = int(cy - yc * lenp)
    ymax = ymin + lenp
    x1 = 0
    x2 = lenp
    y1 = 0
    y2 = lenp
    if xmin < 0:
        x1 = -xmin
        xmin = 0
    if xmax > w:
        x2 = w - (xmax - lenp)
        xmax = w
    if ymin < 0:
        y1 = -ymin
        ymin = 0
    if ymax > h:
        y2 = h - (ymax - lenp)
        ymax = h
    imc = (np.ones((lenp, lenp, 3)) * 255).astype(np.uint8)
    imc[y1:y2, x1:x2, :] = im[ymin:ymax, xmin:xmax, :]
    imr = cv2.resize(imc, (512, 512))
    return imr


def pad_to_square(im):
    h, w, _ = im.shape
    ns = int(max(h, w) * 1.5)
    im = cv2.copyMakeBorder(im, int((ns - h) / 2), (ns - h) - int((ns - h) / 2), int((ns - w) / 2),
                            (ns - w) - int((ns - w) / 2), cv2.BORDER_CONSTANT, 255)
    return im


def post_process_naive(result_list, score_gender, score_age):
    # determine trigger word
    gender = np.argmax(score_gender)
    age = np.argmax(score_age)
    if age < 2:
        if gender == 0:
            tag_a_g = ['a boy', 'children']
        else:
            tag_a_g = ['a girl', 'children']
    elif age > 4:
        if gender == 0:
            tag_a_g = ['a mature man']
        else:
            tag_a_g = ['a mature woman']
    else:
        if gender == 0:
            tag_a_g = ['a handsome man']
        else:
            tag_a_g = ['a beautiful woman']
    num_images = len(result_list)
    cnt_girl = 0
    cnt_boy = 0
    result_list_new = []
    for result in result_list:
        result_new = []
        result_new.extend(tag_a_g)
        ## don't include other infos for lora training
        #for tag in result:
        #    if tag == '1girl' or tag == '1boy':
        #        continue
        #    if tag[-4:] == '_man':
        #        continue
        #    if tag[-6:] == '_woman':
        #        continue
        #    if tag[-5:] == '_male':
        #        continue
        #    elif tag[-7:] == '_female':
        #        continue
        #    elif (
        #            tag == 'ears' or tag == 'head' or tag == 'face' or tag == 'lips' or tag == 'mouth' or tag == '3d' or tag == 'asian' or tag == 'teeth'):
        #        continue
        #    elif ('eye' in tag and not 'eyewear' in tag):
        #        continue
        #    elif ('nose' in tag or 'body' in tag):
        #        continue
        #    elif tag[-5:] == '_lips':
        #        continue
        #    else:
        #        result_new.append(tag)
        #    # import pdb;pdb.set_trace()
        ## result_new.append('slim body')
        result_list_new.append(result_new)

    return result_list_new


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    if s1 < 1.0e-4:
        s1 = 1.0e-4
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])


def rotate(im, keypoints):
    h, w, _ = im.shape
    points_array = np.zeros((5, 2))
    dst_mean_face_size = 160
    dst_mean_face = np.asarray([0.31074522411511746, 0.2798131190011913,
                                0.6892073313037804, 0.2797830232679366,
                                0.49997367716346774, 0.5099309118810921,
                                0.35811903020866753, 0.7233174007629063,
                                0.6418878095835022, 0.7232890570786875])
    dst_mean_face = np.reshape(dst_mean_face, (5, 2)) * dst_mean_face_size

    for k in range(5):
        points_array[k, 0] = keypoints[2 * k]
        points_array[k, 1] = keypoints[2 * k + 1]

    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in points_array]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in dst_mean_face]))
    trans_mat = transformation_from_points(pts1, pts2)
    if trans_mat[1, 1] > 1.0e-4:
        angle = math.atan(trans_mat[1, 0] / trans_mat[1, 1])
    else:
        angle = math.atan(trans_mat[0, 1] / trans_mat[0, 2])
    im = pad_to_square(im)
    ns = int(1.5 * max(h, w))
    M = cv2.getRotationMatrix2D((ns / 2, ns / 2), angle=-angle / np.pi * 180, scale=1.0)
    im = cv2.warpAffine(im, M=M, dsize=(ns, ns))
    return im


def get_mask_head(result):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    mask_hair = np.zeros((512, 512))
    mask_face = np.zeros((512, 512))
    mask_human = np.zeros((512, 512))
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
    mask_head = np.zeros((512, 512)).astype(np.uint8)
    cv2.fillPoly(mask_head, [contours[max_idx]], 255)
    mask_head = mask_head.astype(np.float32) / 255
    mask_head = np.clip(mask_head + mask_face, 0, 1)
    mask_head = np.expand_dims(mask_head, 2)
    return mask_head

def compare_jpg_with_face_id(embedding_list):
    embedding_array = np.vstack(embedding_list)
    # average embedding 
    pivot_feature   = np.mean(embedding_array, axis=0)
    pivot_feature   = np.reshape(pivot_feature, [512, 1])
    # get scores
    scores = [np.dot(emb, pivot_feature)[0][0] for emb in embedding_list]
    return scores

def most_similar_faces_pick(imdir, face_detection, face_recognition, face_quality_func):
    # ---------------------------calculate faceid best jpgs -------------------------- #
    imlist = os.listdir(imdir)

    face_id_scores  = []
    quality_scores  = []
    face_angles     = []
    selected_paths  = []
    for index, jpg in enumerate(tqdm(imlist)):
        try:
            if not jpg.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                continue
            img_path = os.path.join(imdir, jpg)
            image       = Image.open(img_path)
            h, w, c     = np.shape(image)

            retinaface_box, retinaface_keypoint, _ = call_face_crop(face_detection, image, 3, prefix="tmp")
            retinaface_keypoint = np.reshape(retinaface_keypoint, [5, 2])
            # caclculate angle
            x = retinaface_keypoint[0,0] - retinaface_keypoint[1,0]
            y = retinaface_keypoint[0,1] - retinaface_keypoint[1,1]
            angle = 0 if x==0 else abs(math.atan(y/x)*180/math.pi)
            angle = (90 - angle)/ 90 

            # face width
            face_width  = (retinaface_box[2] - retinaface_box[0]) / (3 - 1)
            face_height = (retinaface_box[3] - retinaface_box[1]) / (3 - 1)
            if face_width / w < 1/8 or face_height / h < 1/8:
                continue

            sub_image = image.crop(retinaface_box)

            embedding   = np.array(face_recognition(sub_image)[OutputKeys.IMG_EMBEDDING])
            score       = face_quality_func(sub_image)[OutputKeys.SCORES]
            score       = 0 if score is None else score[0]

            face_id_scores.append(embedding)
            quality_scores.append(score)
            face_angles.append(angle)
            selected_paths.append(jpg)
        except Exception as e:
            print(f'skipping {jpg} due to processing error {e}')

    # sort all scores with muliply
    face_id_scores      = compare_jpg_with_face_id(face_id_scores)
    ref_total_scores    = np.array(face_id_scores) * np.array(quality_scores) * np.array(face_angles)
    ref_indexes         = np.argsort(ref_total_scores)[::-1]

    return ref_total_scores, ref_indexes, selected_paths

class Blipv2():
    def __init__(self):
        self.model = DeepDanbooru()
        self.skin_retouching = pipeline('skin-retouching-torch', model='damo/cv_unet_skin_retouching_torch', model_revision='v1.0.1')
        self.face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd', model_revision='v1.1')
        # self.mog_face_detection_func = pipeline(Tasks.face_detection, 'damo/cv_resnet101_face-detection_cvpr22papermogface')
        self.segmentation_pipeline = pipeline(Tasks.image_segmentation,
                                              'damo/cv_resnet101_image-multiple-human-parsing', model_revision='v1.0.1')
        self.fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition,
                                                 'damo/cv_resnet34_face-attribute-recognition_fairface', model_revision='v2.0.2')
        self.facial_landmark_confidence_func = pipeline(Tasks.face_2d_keypoints,
                                                        'damo/cv_manual_facial-landmark-confidence_flcm', model_revision='v2.5')
        # embedding
        self.face_recognition = pipeline(Tasks.face_recognition, model='damo/cv_ir101_facerecognition_cfglint', model_revision='v1.0.0')
        # face quality
        self.face_quality_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa',  model_revision='v2.0')

    def __call__(self, imdir):
        self.model.start()
        savedir = str(imdir) + '_labeled'
        ensembledir = str(imdir) + '_ensemble'
        shutil.rmtree(savedir, ignore_errors=True)
        shutil.rmtree(ensembledir, ignore_errors=True)
        os.makedirs(savedir, exist_ok=True)
        os.makedirs(ensembledir, exist_ok=True)

        result_list = []
        imgs_list = []

        ref_total_scores, ref_indexes, selected_paths = most_similar_faces_pick(imdir, self.face_detection, self.face_recognition, self.face_quality_func)

        # select top-15 photos for training
        _selected_paths = []
        for index in ref_indexes[:15]:
            _selected_paths.append(selected_paths[index])
            print("selected paths:", selected_paths[index], "total scores: ", ref_total_scores[index])
        
        # copy to the ensembledir for reading
        for i, index in enumerate(ref_indexes[:4]):
            save_path = os.path.join(ensembledir, f"best_roop_image_{str(i)}.jpg")
            os.system(f"cp -rf {os.path.join(imdir, selected_paths[index])} {save_path}")

        cnt = 0
        tmp_path = os.path.join(savedir, 'tmp.png')
        for imname in _selected_paths:
            try:
                # if 1:
                if imname.startswith('.'):
                    continue
                img_path = os.path.join(imdir, imname)
                im = cv2.imread(img_path)
                h, w, _ = im.shape
                max_size = max(w, h)
                ratio = 1024 / max_size
                new_w = round(w * ratio)
                new_h = round(h * ratio)
                imt = cv2.resize(im, (new_w, new_h))
                cv2.imwrite(tmp_path, imt)
                result_det = self.face_detection(tmp_path)
                bboxes = result_det['boxes']
                if len(bboxes) > 1:
                    areas = []
                    for i in range(len(bboxes)):
                        bbox = bboxes[i]
                        areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    areas = np.array(areas)
                    areas_new = np.sort(areas)[::-1]
                    idxs = np.argsort(areas)[::-1]
                    if areas_new[0] < 4 * areas_new[1]:
                        print('Detecting multiple faces, do not use image {}.'.format(imname))
                        continue
                    else:
                        keypoints = result_det['keypoints'][idxs[0]]
                elif len(bboxes) == 0:
                    print('Detecting no face, do not use image {}.'.format(imname))
                    continue
                else:
                    keypoints = result_det['keypoints'][0]

                im = rotate(im, keypoints)
                ns = im.shape[0]
                imt = cv2.resize(im, (1024, 1024))
                cv2.imwrite(tmp_path, imt)
                result_det = self.face_detection(tmp_path)
                bboxes = result_det['boxes']

                if len(bboxes) > 1:
                    areas = []
                    for i in range(len(bboxes)):
                        bbox = bboxes[i]
                        areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    areas = np.array(areas)
                    areas_new = np.sort(areas)[::-1]
                    idxs = np.argsort(areas)[::-1]
                    if areas_new[0] < 4 * areas_new[1]:
                        print('Detecting multiple faces after rotation, do not use image {}.'.format(imname))
                        continue
                    else:
                        bbox = bboxes[idxs[0]]
                elif len(bboxes) == 0:
                    print('Detecting no face after rotation, do not use this image {}'.format(imname))
                    continue
                else:
                    bbox = bboxes[0]

                for idx in range(4):
                    bbox[idx] = bbox[idx] * ns / 1024
                imr = crop_and_resize(im, bbox)
                cv2.imwrite(tmp_path, imr)

                result = self.skin_retouching(tmp_path)
                if (result is None or (result[OutputKeys.OUTPUT_IMG] is None)):
                    print('Cannot do skin retouching, do not use this image.')
                    continue
                cv2.imwrite(tmp_path, result[OutputKeys.OUTPUT_IMG])

                result = self.segmentation_pipeline(tmp_path)
                mask_head = get_mask_head(result)
                im = cv2.imread(tmp_path)
                im = im * mask_head + 255 * (1 - mask_head)
                # print(im.shape)

                raw_result = self.facial_landmark_confidence_func(im)
                if raw_result is None:
                    print('landmark quality fail...')
                    continue

                print(imname, raw_result['scores'][0])
                if float(raw_result['scores'][0]) < (1 - 0.145):
                    print('landmark quality fail...')
                    continue

                cv2.imwrite(os.path.join(savedir, '{}.png'.format(cnt)), im)
                imgs_list.append('{}.png'.format(cnt))
                img = Image.open(os.path.join(savedir, '{}.png'.format(cnt)))
                result = self.model.tag(img)
                print(result)
                attribute_result = self.fair_face_attribute_func(tmp_path)
                if cnt == 0:
                    score_gender = np.array(attribute_result['scores'][0])
                    score_age = np.array(attribute_result['scores'][1])
                else:
                    score_gender += np.array(attribute_result['scores'][0])
                    score_age += np.array(attribute_result['scores'][1])

                result_list.append(result.split(', '))
                cnt += 1
            except Exception as e:
                print('cathed for image process of ' + imname)
                print(f'Error: {e}')

        print(result_list)
        if len(result_list) == 0:
            print('Error: result is empty.')
            exit()
            # return os.path.join(savedir, "metadata.jsonl")

        result_list = post_process_naive(result_list, score_gender, score_age)
        self.model.stop()
        try:
            os.remove(tmp_path)
        except OSError as e:
            print(f"Failed to remove path {tmp_path}: {e}")

        out_json_name = os.path.join(savedir, "metadata.jsonl")
        fo = open(out_json_name, 'w')
        for i in range(len(result_list)):
            generated_text = ", ".join(result_list[i])
            print(imgs_list[i], generated_text)
            info_dict = {"file_name": imgs_list[i], "text": "<fcsks>, " + generated_text}
            fo.write(json.dumps(info_dict) + '\n')
        fo.close()
        return out_json_name
