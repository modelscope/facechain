# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any
import tempfile
from modelscope.pipelines import pipeline
from facechain.constants import tts_speakers_map
try:
    import edge_tts
except ImportError:
    print("警告：未找到edge_tts模块，语音合成功能将无法使用。您可以通过`pip install edge-tts`安装它。\n Warning: The edge_tts module is not found, so the speech synthesis function will not be available. You can install it by 'pip install edge-tts'.")

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


async def text_to_speech_edge(text, speaker):
    voice = tts_speakers_map[speaker]
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name

    await communicate.save(tmp_path)

    return tmp_path