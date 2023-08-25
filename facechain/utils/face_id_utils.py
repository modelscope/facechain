import base64
import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import requests

try:
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
except:
    print("modelscope is not installed. If local faceid is used, please install model scope or use faceid_post_url")
    pass


def decode_image_from_base64jpeg(base64_image):
    image_bytes = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def post_face_get_emb(img_path, faceid_post_url='http://0.0.0.0:8005/test'):
    with open(img_path, 'rb') as f:
        encoded_image   = base64.b64encode(f.read()).decode('utf-8')
        datas           = json.dumps({
            'image': encoded_image,
            'only_embedding':True
        })

        r       = requests.post(faceid_post_url, data=datas, timeout=1500)
        outputs = r.content.decode('utf-8')
        outputs = json.loads(outputs)
        for key in outputs.keys():
            if 'embedding' in key:
                emb = np.array(outputs[key])
    return emb

def eval_jpg_with_faceidremote(pivot_dir, test_img_dir, top_merge=10, faceid_post_url='http://0.0.0.0:8005/test'):
    """
    Evaluate images using remote face identification.

    Args:
        pivot_dir (str): Directory containing reference real human images.
        test_img_dir (str): Directory pointing to generated validation images for training.
            Image names follow the format xxxx_{step}_{indx}.jpg.
        top_merge (int, optional): Number of top weights to select for merging. Defaults to 10.
        faceid_post_url (str, optional): URL address for posting faceid data. Defaults to 'http://0.0.0.0:8005/test'.

    Returns:
        list: List of evaluated results.

    Function:
        - Obtain faceid features through a post request.
        - Calculate the average feature of real human images.
        - Select top_merge weights for merging based on generated validation images.
    """

    # Get the list of real human images

    face_image_list = glob(os.path.join(pivot_dir, '*.jpg')) + glob(os.path.join(pivot_dir, '*.JPG')) + \
                      glob(os.path.join(pivot_dir, '*.png')) + glob(os.path.join(pivot_dir, '*.PNG'))
    embedding_list = []

    # vstack all embedding
    for img in face_image_list:
        try :
            embedding_list.append(post_face_get_emb(img, faceid_post_url=faceid_post_url))
        except:
            pass
    embedding_array = np.vstack(embedding_list)
    
    # mean get pivot of ID
    pivot_feature   = np.mean(embedding_array, axis=0)
    pivot_feature   = np.reshape(pivot_feature, [512, 1])

    # sort by cosine distance
    embedding_list  = [[np.dot(emb,pivot_feature)[0][0], emb] for emb in embedding_list]
    embedding_list  = sorted(embedding_list, key = lambda a : -a[0])
    
    
    top10_embedding         = [emb[1] for emb in embedding_list]
    top10_embedding_array   = np.vstack(top10_embedding)
    
    # [512, n]
    top10_embedding_array   = np.swapaxes(top10_embedding_array, 0, 1)
    print('pivot features : ', top10_embedding_array.shape)

    # traverse through the generated validation images for training, calculate scores, and sort them
    result_list = []
    if not test_img_dir.endswith('.jpg'):
        img_list = glob(os.path.join(test_img_dir, '*.jpg'))
        for img in img_list:
            try:
                emb1        = post_face_get_emb(img)
                res         = np.mean(np.dot(emb1, top10_embedding_array))
                result_list.append([res, img])
                result_list = sorted(result_list, key = lambda a : -a[0])
            except:
                pass
  
    t_result_list = [i[1] for i in result_list][:top_merge]
    tlist   = [i[1].split('_')[-2] for i in result_list][:top_merge]
    scores  = [i[0] for i in result_list][:top_merge]
    return t_result_list, tlist, scores

def eval_jpg_with_faceid(pivot_dir, test_img_dir, top_merge=10):
    """
    Evaluate images using local face identification.

    Args:
        pivot_dir (str): Directory containing reference real human images.
        test_img_dir (str): Directory pointing to generated validation images for training.
            Image names follow the format xxxx_{step}_{indx}.jpg.
        top_merge (int, optional): Number of top weights to select for merging. Defaults to 10.

    Returns:
        list: List of evaluated results.

    Function:
        - Obtain faceid features locally.
        - Calculate the average feature of real human images.
        - Select top_merge weights for merging based on generated validation images.
    """

    # Create a face_recognition model

    face_recognition    = pipeline(Tasks.face_recognition, model='damo/cv_ir101_facerecognition_cfglint')
    # get ID list
    face_image_list     = glob(os.path.join(pivot_dir, '*.jpg')) + glob(os.path.join(pivot_dir, '*.JPG')) + \
                          glob(os.path.join(pivot_dir, '*.png')) + glob(os.path.join(pivot_dir, '*.PNG'))
    
    #  vstack all embedding
    embedding_list = []
    for img in face_image_list:
        embedding_list.append(face_recognition(img)[OutputKeys.IMG_EMBEDDING])
    embedding_array = np.vstack(embedding_list)
    
    #  mean, get pivot
    pivot_feature   = np.mean(embedding_array, axis=0)
    pivot_feature   = np.reshape(pivot_feature, [512, 1])

    # sort with cosine distance
    embedding_list = [[np.dot(emb, pivot_feature)[0][0], emb] for emb in embedding_list]
    embedding_list = sorted(embedding_list, key = lambda a : -a[0])
    # for i in range(10):
    #     print(embedding_list[i][0], embedding_list[i][1].shape)

    top10_embedding         = [emb[1] for emb in embedding_list]
    top10_embedding_array   = np.vstack(top10_embedding)
    # [512, n]
    top10_embedding_array   = np.swapaxes(top10_embedding_array, 0, 1)

    # sort all validation image
    result_list = []
    if not test_img_dir.endswith('.jpg'):
        img_list = glob(os.path.join(test_img_dir, '*.jpg')) + glob(os.path.join(test_img_dir, '*.JPG')) + \
                   glob(os.path.join(test_img_dir, '*.png')) + glob(os.path.join(test_img_dir, '*.PNG'))
        for img in img_list:
            try:
                # a average above all
                emb1 = face_recognition(img)[OutputKeys.IMG_EMBEDDING]
                res = np.mean(np.dot(emb1, top10_embedding_array))
                result_list.append([res, img])
                result_list = sorted(result_list, key = lambda a : -a[0])
            except:
                pass

    # pick most similar using faceid
    t_result_list = [i[1] for i in result_list][:top_merge]
    tlist   = [i[1].split('_')[-2] for i in result_list][:top_merge]
    scores  = [i[0] for i in result_list][:top_merge]
    return t_result_list, tlist, scores

if __name__=="__main__":
    pivot_dir       = sys.argv[1]
    test_img_dir    = sys.argv[2]
    top_merge = 5
    # return_res  = eval_jpg_with_faceid(pivot_dir, test_img_dir)
    return_res      = eval_jpg_with_faceidremote(pivot_dir, test_img_dir)