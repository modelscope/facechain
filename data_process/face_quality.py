import argparse
import os 
import cv2
import numpy as np
import time
import sys
import torch
import onnxruntime
import onnx

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_lst', type=str, default='./multi_det/single_face.txt', help='List for single Face images')
    parser.add_argument('--quality_threshold', type=float, default=0.75, help='The threshold for quality select')
    parser.add_argument('--output_file', type=str, default='./valid_data.txt', help='The threshold for quality select')
    parser.add_argument('--low_output_file', type=str, default='./quality_low.txt', help='The threshold for quality select')
    
    
    args = parser.parse_args() 
    return args

def to_rgb(img):
    h, w = img.shape
    ret = np.empty((h, w, 3), dtype = np.uint8)
    ret[:,:,0] = ret[:,:,1] = ret[:,:,2] = img
    return ret

def main(args):
    input_file = open(args.input_lst, 'r')
    model_path = 'quality_v2_bgr_10_out_tiny_112_45_128.onnx'
    model = onnx.load(model_path)
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    fileout = open(args.output_file, 'w')
    lowout = open(args.low_output_file, 'w')
    for line in input_file:
        line_vec = line.strip().split()
        img_path = os.path.join('./aligned_masked' , line_vec[0].split('/')[-1])
        img = cv2.imread(img_path)
        if img is None:
            print('cannot find ', img_path)
            continue
        if img.ndim == 2:
            img = to_rgb(img)
        img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32)
        img = (img - 127.5) * 0.0078125
        img = np.expand_dims(img, 0).copy()
        img_tensor = torch.from_numpy(img)
        img_nchw = img_tensor.permute(0, 3, 1, 2)
        img_nchw_np = img_nchw.numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: img_nchw_np}
        outputs = ort_session.run(None, ort_inputs)
        quality_score = np.mean(outputs)
        if quality_score >= args.quality_threshold:
            fileout.write(img_path + ' ' + str(quality_score)  + '\n')
            print(img_path, ' has been qualitied/n')
        else:
            lowout.write(img_path + ' ' + str(quality_score) + '\n')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)



        


        

