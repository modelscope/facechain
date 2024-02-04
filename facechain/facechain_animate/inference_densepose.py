import torch
import datetime
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer
import tempfile
import shutil
from facechain.utils import snapshot_download
from facechain.utils import join_worker_data_dir
import os
import subprocess

class DensePose():
    def __init__(self, uuid, config="facechain_animate/densepose/configs/densepose_rcnn_R_50_FPN_s1x.yaml") -> None:
        if not uuid:
            if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
                return "请登陆后使用! (Please login first)"
            else:
                uuid = 'qw'
        self.save_dir = join_worker_data_dir(uuid, 'animate')

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file("facechain_animate/densepose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        cfg.MODEL.WEIGHTS = os.path.join(snapshot_download('eavesy/vid2densepose'), "model_final_162be9.pkl")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg


    def __call__(self, input_video_path):
        predictor = DefaultPredictor(self.cfg)
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = os.path.join(self.save_dir,'densepose')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        densepose_path = f"{savedir}/{time_str}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'h264')
        out = cv2.VideoWriter(densepose_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            with torch.no_grad():
                outputs = predictor(frame)['instances']
            
            results = DensePoseResultExtractor()(outputs)
            cmap = cv2.COLORMAP_VIRIDIS
            # Visualizer outputs black for background, but we want the 0 value of
            # the colormap, so we initialize the array with that value
            arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
            out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)        
            out.write(out_frame)

        cap.release()
        out.release()
        return densepose_path