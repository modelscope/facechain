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
        pivot_dir       参考的真人图片，指向一个文件夹；
        test_img_dir    指向训练生成的validation图像，图像名称满足xxxx_{step}_{indx}.jpg
        top_merge       选择top_merge个权重进行merge，权重可重复使用；
        faceid_post_url 指向post的url地址。

        函数功能：
        faceid特征通过post获得
        获取真人图片的平均特征，然后根据训练生成的validation图像，选择top_merge个权重进行merge。
    """
    # 获取真人图像的列表
    face_image_list = glob(os.path.join(pivot_dir, '*.jpg')) + glob(os.path.join(pivot_dir, '*.JPG')) + \
                      glob(os.path.join(pivot_dir, '*.png')) + glob(os.path.join(pivot_dir, '*.PNG'))
    embedding_list = []

    # 获取每张真人图像的embedding后进行堆叠
    for img in face_image_list:
        try :
            embedding_list.append(post_face_get_emb(img, faceid_post_url=faceid_post_url))
        except:
            pass
    embedding_array = np.vstack(embedding_list)
    # 然后对真人图片取mean，获取真人图片的平均特征
    pivot_feature   = np.mean(embedding_array, axis=0)
    pivot_feature   = np.reshape(pivot_feature, [512, 1])

    # 计算一个文件夹中，和中位值最接近的图片排序
    embedding_list  = [[np.dot(emb,pivot_feature)[0][0], emb] for emb in embedding_list]
    embedding_list  = sorted(embedding_list, key = lambda a : -a[0])
    
    # 取出embedding
    top10_embedding         = [emb[1] for emb in embedding_list]
    top10_embedding_array   = np.vstack(top10_embedding)
    # [512, n]
    top10_embedding_array   = np.swapaxes(top10_embedding_array, 0, 1)
    print('pivot features : ', top10_embedding_array.shape)

    # 遍历训练生成的validation图像，并且计算得分，并排序
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
    # 最相似的几张图片
    t_result_list = [i[1] for i in result_list][:top_merge]
    # i[1].split('_')[-2]代表的是第n步
    tlist   = [i[1].split('_')[-2] for i in result_list][:top_merge]
    # i[0]代表的是第n步的得分
    scores  = [i[0] for i in result_list][:top_merge]
    return t_result_list, tlist, scores

def eval_jpg_with_faceid(pivot_dir, test_img_dir, top_merge=10):
    """
        pivot_dir       参考的真人图片，指向一个文件夹；
        test_img_dir    指向训练生成的validation图像，图像名称满足xxxx_{step}_{indx}.jpg
        top_merge       选择top_merge个权重进行merge，权重可重复使用；

        函数功能：
        faceid特征通过本地获得
        获取真人图片的平均特征，然后根据训练生成的validation图像，选择top_merge个权重进行merge。
    """
    # 创建face_recognition模型
    face_recognition    = pipeline(Tasks.face_recognition, model='damo/cv_ir101_facerecognition_cfglint')
    # 获取真人图像的列表
    face_image_list     = glob(os.path.join(pivot_dir, '*.jpg')) + glob(os.path.join(pivot_dir, '*.JPG')) + \
                          glob(os.path.join(pivot_dir, '*.png')) + glob(os.path.join(pivot_dir, '*.PNG'))
    
    # 获取每张真人图像的embedding后进行堆叠
    embedding_list = []
    for img in face_image_list:
        embedding_list.append(face_recognition(img)[OutputKeys.IMG_EMBEDDING])
    embedding_array = np.vstack(embedding_list)
    # 然后对真人图片取mean，获取真人图片的平均特征
    pivot_feature   = np.mean(embedding_array, axis=0)
    pivot_feature   = np.reshape(pivot_feature, [512, 1])

    # 计算一个文件夹中，和中位值最接近的图片排序
    embedding_list = [[np.dot(emb, pivot_feature)[0][0], emb] for emb in embedding_list]
    embedding_list = sorted(embedding_list, key = lambda a : -a[0])
    # for i in range(10):
    #     print(embedding_list[i][0], embedding_list[i][1].shape)
    
    # 取出embedding
    top10_embedding         = [emb[1] for emb in embedding_list]
    top10_embedding_array   = np.vstack(top10_embedding)
    # [512, n]
    top10_embedding_array   = np.swapaxes(top10_embedding_array, 0, 1)

    # 遍历训练生成的validation图像，并且计算得分，并排序
    result_list = []
    if not test_img_dir.endswith('.jpg'):
        img_list = glob(os.path.join(test_img_dir, '*.jpg')) + glob(os.path.join(test_img_dir, '*.JPG')) + \
                   glob(os.path.join(test_img_dir, '*.png')) + glob(os.path.join(test_img_dir, '*.PNG'))
        for img in img_list:
            try:
                # 生成人脸和所有真实人脸的平均得分
                emb1 = face_recognition(img)[OutputKeys.IMG_EMBEDDING]
                res = np.mean(np.dot(emb1, top10_embedding_array))
                result_list.append([res, img])
                result_list = sorted(result_list, key = lambda a : -a[0])
            except:
                pass
    # 最相似的几张图片
    t_result_list = [i[1] for i in result_list][:top_merge]
    # i[1].split('_')[-2]代表的是第n步
    tlist   = [i[1].split('_')[-2] for i in result_list][:top_merge]
    # i[0]代表的是第n步的得分
    scores  = [i[0] for i in result_list][:top_merge]
    return t_result_list, tlist, scores

if __name__=="__main__":
    pivot_dir       = sys.argv[1]
    test_img_dir    = sys.argv[2]
    top_merge = 5
    # return_res  = eval_jpg_with_faceid(pivot_dir, test_img_dir)
    return_res      = eval_jpg_with_faceidremote(pivot_dir, test_img_dir)