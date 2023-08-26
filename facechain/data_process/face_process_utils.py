import cv2
import numpy as np
import os
from PIL import Image
from skimage import transform

def safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, face_seg, mask_type):
    """
    Get expanded box, keypoints, and face segmentation mask.

    Args:
        image (np.ndarray): The input image.
        retinaface_result (dict): The detection result from RetinaFace.
        crop_ratio (float): The ratio for expanding the crop of the face part.
        face_seg (function): The face segmentation model.
        mask_type (str): The method of face segmentation, either 'crop' or 'skin'.

    Returns:
        np.ndarray: The box relative to the original image, expanded.
        np.ndarray: The keypoints relative to the original image.
        Image: The face segmentation result.
    """
    h, w, c = np.shape(image)
    if len(retinaface_result['boxes']) != 0:
        # Get the RetinaFace box and expand it
        retinaface_box = np.array(retinaface_result['boxes'][0])
        face_width = retinaface_box[2] - retinaface_box[0]
        face_height = retinaface_box[3] - retinaface_box[1]
        retinaface_box[0] = np.clip(retinaface_box[0] - face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[1] = np.clip(retinaface_box[1] - face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box[2] = np.clip(retinaface_box[2] + face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[3] = np.clip(retinaface_box[3] + face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box = np.array(retinaface_box, np.int32)

        # Detect keypoints
        retinaface_keypoints = np.reshape(retinaface_result['keypoints'][0], [5, 2])
        retinaface_keypoints = np.array(retinaface_keypoints, np.float32)

        # Mask part
        # retinaface_crop = Image.fromarray(image).crop(tuple(np.int32(retinaface_box)))
        retinaface_crop = image.crop(tuple(np.int32(retinaface_box)))
        retinaface_mask = np.zeros_like(np.array(image, np.uint8))
        if mask_type == "skin":
            retinaface_sub_mask = face_seg(retinaface_crop)
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = np.expand_dims(retinaface_sub_mask, -1)
        else:
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))
    else:
        retinaface_box = np.array([])
        retinaface_keypoints = np.array([])
        retinaface_mask = np.zeros_like(np.array(image, np.uint8))
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))

    return retinaface_box, retinaface_keypoints, retinaface_mask_pil


def crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box):
    """
    Crop and paste a face from the source image to the target image.

    Args:
        Source_image (Image): The source image.
        Source_image_mask (Image): The mask of the face in the source image.
        Target_image (Image): The target template image.
        Source_Five_Point (np.ndarray): Five facial keypoints in the source image.
        Target_Five_Point (np.ndarray): Five facial keypoints in the target image.
        Source_box (list): The coordinates of the face box in the source image.

    Returns:
        np.ndarray: The output image with the face pasted.
    """
    Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(Source_box[:2])
    Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

    Crop_Source_image                       = Source_image.crop(np.int32(Source_box))
    Crop_Source_image_mask                  = Source_image_mask.crop(np.int32(Source_box))
    Source_Five_Point, Target_Five_Point    = np.array(Source_Five_Point), np.array(Target_Five_Point)

    tform = transform.SimilarityTransform()
    tform.estimate(Source_Five_Point, Target_Five_Point)
    M = tform.params[0:2, :]

    warped      = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)
    warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)

    mask        = np.float32(warped_mask == 0)
    output      = mask * np.float32(Target_image) + (1 - mask) * np.float32(warped)
    return output


def call_face_crop(retinaface_detection, image, crop_ratio, prefix="tmp"):
    """
    Perform face detection, mask, and keypoint extraction using RetinaFace.

    Args:
        retinaface_detection (function): The RetinaFace detection function.
        image (Image): The input image.
        crop_ratio (float): The crop ratio for face expansion.
        prefix (str): Prefix for temporary files (default is "tmp").

    Returns:
        np.ndarray: Detected face bounding box.
        np.ndarray: Detected face keypoints.
        Image: Extracted face mask.
    """
    # Perform RetinaFace detection
    retinaface_result = retinaface_detection(image)
    
    # Get mask and keypoints
    retinaface_box, retinaface_keypoints, retinaface_mask_pil = safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, None, "crop")
    
    return retinaface_box, retinaface_keypoints, retinaface_mask_pil