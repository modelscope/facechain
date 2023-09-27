# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import sys

from typing import Any


class SadTalker():
    def __init__(self, uuid):
        if not uuid:
            if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
                return "请登陆后使用! (Please login first)"
            else:
                uuid = 'qw'

        # self.save_dir = os.path.join('/tmp', uuid, 'sythesized_video') # deprecated
        self.save_dir = os.path.join('.', uuid, 'sythesized_video')

    def __call__(self, *args, **kwargs) -> Any:
        # two required arguments
        source_image = kwargs.get("source_image") or args[0]
        driven_audio = kwargs.get('driven_audio') or args[1]
        # other optional arguments
        kwargs = {
            'preprocess' : kwargs.get('preprocess') or args[2], 
            'still_mode' : kwargs.get('still_mode') or args[3],
            'use_enhancer' : kwargs.get('use_enhancer') or args[4],
            'batch_size' : kwargs.get('batch_size') or args[5],
            'size' : kwargs.get('size') or args[6], 
            'pose_style' : kwargs.get('pose_style') or args[7],
            'exp_scale' : kwargs.get('exp_scale') or args[8],
            'result_dir': self.save_dir
        }
        inference = pipeline('talking-head', model='wwd123/sadtalker', model_revision='v1.0.0')
        print("initialized sadtalker pipeline")
        video_path = inference(source_image, driven_audio=driven_audio, **kwargs)
        return video_path