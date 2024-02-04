import os
import errno
import sys
import argparse
import numpy as np
import face_preprocess
import cv2
import re
import json

def to_rgb(img):
    h, w = img.shape
    ret = np.empty((h, w, 3), dtype = np.uint8)
    ret[:,:,0] = ret[:,:,1] = ret[:,:,2] = img
    return ret

def main(args):
    input_file = open(args.input_lst, 'r')
    for line in input_file:
        line_vec = line.strip().split(' ', 1)
        imgpath = line_vec[0]
        imgname = os.path.split(imgpath)[-1]
        img = cv2.imread(imgpath)
        if img is None:
            print('cannot find ', imgpath)
            continue
        if img.ndim == 2:
            img = to_rgb(img)
        mask = cv2.imread(os.path.join('./pure_face_mask', imgname))
        if mask is None:
            mask = np.ones((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        mask = mask.astype(np.float32) / 255.
        img_masked = np.clip((mask * img).astype(np.uint8), 0, 255)
        rect_points_vec = line_vec[-1].split(' ')
        # print(len(rect_points_vec))
        assert len(rect_points_vec) == 15
        # align
        points_vec = [float(e) for e in rect_points_vec[4:-1]]
        _landmark = np.array(points_vec)
        _landmark = _landmark.reshape(5,2)
        warped = face_preprocess.preprocess(img, bbox=None, landmark = _landmark, image_size = args.image_size)
        warped_masked = face_preprocess.preprocess(img_masked, bbox=None, landmark = _landmark, image_size = args.image_size)
        out_subdir = 'aligned'
        masked_subdir = 'aligned_masked'
        if not os.path.exists(masked_subdir):
            os.makedirs(masked_subdir)
        masked_path = os.path.join(masked_subdir, imgname.rsplit('/', 1)[-1].rsplit('.', 1)[0]+'.jpg')
        cv2.imwrite(masked_path, warped_masked)
        if not os.path.exists(out_subdir):
            try:
                os.makedirs(out_subdir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        if not os.path.exists(out_subdir):
            print(out_subdir + " not exists")
            print(line.strip())
            raise
        out_path = os.path.join(out_subdir, imgname.rsplit('/', 1)[-1].rsplit('.', 1)[0]+'.jpg')
        cv2.imwrite(out_path, warped)
        # crop
        expand_ratio = 1.5
        rect_vec = [int(float(e)) for e in rect_points_vec[:4]]
        h, w, _ = img.shape
        img_edge = min(w, h)
        rect_cx = (rect_vec[0]+rect_vec[2]) // 2
        rect_cy = (rect_vec[1]+rect_vec[3]) // 2

        rect_edge = max(rect_vec[2]-rect_vec[0], rect_vec[3]-rect_vec[1])
        newrect_edge = min(int(expand_ratio * rect_edge), img_edge)
        crop_size = newrect_edge
        xnew_start = rect_cx - crop_size // 2
        ynew_start = rect_cy - crop_size // 2
        xnew_end = xnew_start + crop_size
        ynew_end = ynew_start + crop_size
        if xnew_start < 0:
            xnew_start = 0
            xnew_end = crop_size
        if ynew_start < 0:
            ynew_start = 0
            ynew_end = crop_size
        if xnew_end > w:
            xnew_end = w
            xnew_start = xnew_end - crop_size
        if ynew_end > h:
            ynew_end = h
            ynew_start = ynew_end - crop_size

        assert xnew_start >= 0
        assert ynew_start >= 0
        assert xnew_end <= w
        assert ynew_end <= h
        imgcrop = img[ynew_start:ynew_end,xnew_start:xnew_end,:]
        crop_out_subdir = 'cropimg'
        if not os.path.exists(crop_out_subdir):
            try:
                os.makedirs(crop_out_subdir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        if not os.path.exists(crop_out_subdir):
            print(crop_out_subdir + " not exists")
            print(line.strip())
            raise
        crop_out_path = os.path.join(crop_out_subdir, imgname.rsplit('/', 1)[-1].rsplit('.', 1)[0]+'.jpg')
        cv2.imwrite(crop_out_path, imgcrop)
        print(crop_out_path, 'has been croped \n')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-lst', type = str, default = './det_res/det_res.txt', help = 'List for unaligned images')
    parser.add_argument('--image-size', type = str, default = '112,112', help = 'Image size (height, width) in pixels.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
