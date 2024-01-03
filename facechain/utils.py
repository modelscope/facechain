# Copyright (c) Alibaba, Inc. and its affiliates.

import time
import subprocess
from modelscope import snapshot_download as ms_snapshot_download
import multiprocessing as mp
import os
import numpy as np
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AdaptiveKLController:
    """
    copied from: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.init_kl_coef = init_kl_coef
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * 0.1
        self.value *= mult
    def reset(self):
        self.value = self.init_kl_coef

def max_retries(max_attempts):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Retry {attempts}/{max_attempts}: {e}")
                    # wait 1 sec
                    time.sleep(1)
            raise Exception(f"Max retries ({max_attempts}) exceeded.")
        return wrapper
    return decorator


@max_retries(3)
def snapshot_download(*args, **kwargs):
    return ms_snapshot_download(*args, **kwargs)


def pre_download_models():
    snapshot_download('ly261666/cv_portrait_model', revision='v4.0')
    snapshot_download('YorickHe/majicmixRealistic_v6', revision='v1.0.0')
    snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
    snapshot_download('ly261666/cv_wanx_style_model', revision='v1.0.3')
    snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
    snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
    snapshot_download('Cherrytest/rot_bgr', revision='v1.0.0')
    snapshot_download('damo/face_frombase_c4', revision='v1.0.0')


def set_spawn_method():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("spawn method already set")

def check_install(*args):
    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT)
        return True
    except OSError as e:
        return False

def check_ffmpeg():
    """
    Check if ffmpeg is installed.
    """
    return check_install("ffmpeg", "-version")


def get_worker_data_dir() -> str:
    """
    Get the worker data directory.
    """
    return os.path.join(project_dir, "worker_data")


def join_worker_data_dir(*kwargs) -> str:
    """
    Join the worker data directory with the specified sub directory.
    """
    return os.path.join(get_worker_data_dir(), *kwargs)
