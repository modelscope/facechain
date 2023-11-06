# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(pose, H, W, include_body, include_hand, include_face):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    
    if include_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

    if include_hand:
        canvas = util.draw_handpose(canvas, hands)

    if include_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self, model_dir):

        self.pose_estimation = Wholebody(model_dir)

    def __call__(self, oriImg, include_body=True, include_face=True, include_hand=True, return_handbox=False):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
            if return_handbox:
                handbox = []
                for hand in hands:
                    hand_pts = np.array(hand)
                    hand_idxs = (hand_pts > 0) * (hand_pts < 1)
                    hand_pts[:,0] = hand_pts[:,0] * W
                    hand_pts[:,1] = hand_pts[:,1] * H
                    minx, miny = np.min(hand_pts * hand_idxs + 10000 * (1 - hand_idxs), axis=0)
                    maxx, maxy = np.max(hand_pts * hand_idxs, axis=0)
                    if (maxx-minx) * (maxy-miny) > 0 and np.sum(hand_idxs) > 0:
                        w = maxx - minx
                        h = maxy - miny
                        handbox.append([max(0, int(minx-0.5*w)), max(0, int(miny-0.5*h)), min(W, int(maxx+0.5*w)), min(H, int(maxy+0.5*h))])
                return draw_pose(pose, H, W, include_body, include_hand, include_face), handbox
            else:            
                return draw_pose(pose, H, W, include_body, include_hand, include_face)
