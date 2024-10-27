import cv2
import numpy as np
from skimage import transform as trans
from typing import List, Union, Optional

def read_image(img_path: str, mode: str = 'rgb', layout: str = 'HWC') -> np.ndarray:
    """
    Read and process the image based on the specified mode and layout.

    :param img_path: Path to the image file.
    :param mode: Image mode ('rgb' or 'gray').
    :param layout: Image layout ('HWC' or 'CHW').
    :return: Processed image.
    """
    if mode == 'gray':
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if mode == 'rgb':
            img = img[..., ::-1]
        if layout == 'CHW':
            img = np.transpose(img, (2, 0, 1))
    return img

def preprocess(img: Union[str, np.ndarray], bbox: Optional[List[int]] = None, landmark: Optional[np.ndarray] = None,
               image_size: Optional[List[int]] = None, margin: int = 44, mode: str = 'rgb', layout: str = 'HWC') -> np.ndarray:
    """
    Preprocess the image by resizing and aligning based on the bounding box and landmark.

    :param img: Input image or image path.
    :param bbox: Bounding box coordinates.
    :param landmark: Landmark points.
    :param image_size: Desired image size.
    :param margin: Margin for bounding box.
    :param mode: Image mode ('rgb' or 'gray').
    :param layout: Image layout ('HWC' or 'CHW').
    :return: Preprocessed image.
    """
    if isinstance(img, str):
        img = read_image(img, mode=mode, layout=layout)

    image_size = image_size or [112, 112]

    if landmark is not None:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)

        if image_size[1] == 112:
            src[:, 0] += 8.0

        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
    else:
        M = None

    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped

def preprocess_3pt(img: Union[str, np.ndarray], bbox: Optional[List[int]] = None, landmark: Optional[np.ndarray] = None,
                   image_size: Optional[List[int]] = None, margin: int = 44, mode: str = 'rgb', layout: str = 'HWC') -> np.ndarray:
    """
    Preprocess the image by resizing and aligning based on the bounding box and landmark using 3 points.

    :param img: Input image or image path.
    :param bbox: Bounding box coordinates.
    :param landmark: Landmark points.
    :param image_size: Desired image size.
    :param margin: Margin for bounding box.
    :param mode: Image mode ('rgb' or 'gray').
    :param layout: Image layout ('HWC' or 'CHW').
    :return: Preprocessed image.
    """
    if isinstance(img, str):
        img = read_image(img, mode=mode, layout=layout)

    image_size = image_size or [112, 112]

    if landmark is not None:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366]
        ], dtype=np.float32)

        if image_size[1] == 112:
            src[:, 0] += 8.0

        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
    else:
        M = None

    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped
