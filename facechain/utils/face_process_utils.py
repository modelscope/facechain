import cv2
import numpy as np
import os
from PIL import Image
from skimage import transform


def safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, face_seg, mask_type):
    '''
    Inputs:
        image                   输入图片；
        retinaface_result       retinaface的检测结果；
        crop_ratio              人脸部分裁剪扩充比例；
        face_seg                人脸分割模型；
        mask_type               人脸分割的方式，一个是crop，一个是skin，人脸分割结果是人脸皮肤或者人脸框
    
    Outputs:
        retinaface_box          扩增后相对于原图的box
        retinaface_keypoints    相对于原图的keypoints
        retinaface_mask_pil     人脸分割结果
    '''
    h, w, c = np.shape(image)
    if len(retinaface_result['boxes']) != 0:
        # 获得retinaface的box并且做一手扩增
        retinaface_box      = np.array(retinaface_result['boxes'][0])
        face_width          = retinaface_box[2] - retinaface_box[0]
        face_height         = retinaface_box[3] - retinaface_box[1]
        retinaface_box[0]   = np.clip(np.array(retinaface_box[0], np.int32) - face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[1]   = np.clip(np.array(retinaface_box[1], np.int32) - face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box[2]   = np.clip(np.array(retinaface_box[2], np.int32) + face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[3]   = np.clip(np.array(retinaface_box[3], np.int32) + face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box      = np.array(retinaface_box, np.int32)

        # 检测关键点
        retinaface_keypoints = np.reshape(retinaface_result['keypoints'][0], [5, 2])
        retinaface_keypoints = np.array(retinaface_keypoints, np.float32)

        # mask部分
        retinaface_crop     = image.crop(np.int32(retinaface_box))
        retinaface_mask     = np.zeros_like(np.array(image, np.uint8))
        if mask_type == "skin":
            retinaface_sub_mask = face_seg(retinaface_crop)
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = np.expand_dims(retinaface_sub_mask, -1)
        else:
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))
    else:
        retinaface_box          = np.array([])
        retinaface_keypoints    = np.array([])
        retinaface_mask         = np.zeros_like(np.array(image, np.uint8))
        retinaface_mask_pil     = Image.fromarray(np.uint8(retinaface_mask))
        
    return retinaface_box, retinaface_keypoints, retinaface_mask_pil

def crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box):
    '''
    Inputs:
        Source_image            原图像；
        Source_image_mask       原图像人脸的mask比例；
        Target_image            目标模板图像；
        Source_Five_Point       原图像五个人脸关键点；
        Target_Five_Point       目标图像五个人脸关键点；
        Source_box              原图像人脸的坐标；
    
    Outputs:
        output                  贴脸后的人像
    '''
    Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(Source_box[:2])
    Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

    Crop_Source_image                       = Source_image.crop(np.int32(Source_box))
    Crop_Source_image_mask                  = Source_image_mask.crop(np.int32(Source_box))
    Source_Five_Point, Target_Five_Point    = np.array(Source_Five_Point), np.array(Target_Five_Point)

    tform = transform.SimilarityTransform()
    # 程序直接估算出转换矩阵M
    tform.estimate(Source_Five_Point, Target_Five_Point)
    M = tform.params[0:2, :]

    warped      = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)
    warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)

    mask        = np.float32(warped_mask == 0)
    output      = mask * np.float32(Target_image) + (1 - mask) * np.float32(warped)
    return output

def call_face_crop(retinaface_detection, image, crop_ratio, prefix="tmp"):
    # retinaface检测部分
    # 检测人脸框
    retinaface_result                                           = retinaface_detection(image) 
    # 获取mask与关键点
    retinaface_box, retinaface_keypoints, retinaface_mask_pil   = safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, None, "crop")

    return retinaface_box, retinaface_keypoints, retinaface_mask_pil