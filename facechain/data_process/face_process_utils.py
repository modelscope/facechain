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


def crop_and_paste(source_image, source_image_mask, target_image, source_five_point, target_five_point, source_box):
    """
    Crop and paste a face from the source image onto the target image.

    Args:
        source_image (PIL.Image): Original image.
        source_image_mask (PIL.Image): Mask of the face in the original image.
        target_image (PIL.Image): Target template image.
        source_five_point (list): List of five facial keypoints of the source image.
        target_five_point (list): List of five facial keypoints of the target image.
        source_box (tuple): Coordinates of the face region in the source image.
    Returns:
        PIL.Image: Resultant image after face pasting.
    """

    source_five_point = np.reshape(source_five_point, [5, 2]) - np.array(source_box[:2])
    target_five_point = np.reshape(target_five_point, [5, 2])

    crop_source_image = source_image.crop(np.int32(source_box))
    crop_source_image_mask = source_image_mask.crop(np.int32(source_box))
    source_five_point, target_five_point = np.array(source_five_point), np.array(target_five_point)

    tform = transform.SimilarityTransform()
    tform.estimate(source_five_point, target_five_point)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(np.array(crop_source_image), M, np.shape(target_image)[:2][::-1], borderValue=0.0)
    warped_mask = cv2.warpAffine(np.array(crop_source_image_mask), M, np.shape(target_image)[:2][::-1], borderValue=0.0)

    mask = np.float32(warped_mask == 0)
    output = mask * np.float32(target_image) + (1 - mask) * np.float32(warped)
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