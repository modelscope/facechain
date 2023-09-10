# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from modelscope import snapshot_download as ms_snapshot_download


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
