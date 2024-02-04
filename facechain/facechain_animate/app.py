# Copyright (c) Alibaba, Inc. and its affiliates.
import enum
import os
import json
import shutil
import slugify
import time
from concurrent.futures import ProcessPoolExecutor
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import imageio
import torch
from glob import glob
import platform
from facechain.utils import snapshot_download, check_ffmpeg, set_spawn_method, project_dir, join_worker_data_dir

from facechain_animate.inference_animate import MagicAnimate
from facechain_animate.inference_densepose import DensePose
import tempfile

training_done_count = 0
inference_done_count = 0


def get_selected_video(state_video_list, evt: gr.SelectData):
    return state_video_list[evt.index]

def get_previous_video_result(uuid):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            return "请登陆后使用! (Please login first)"
        else:
            uuid = 'qw'

    save_dir = join_worker_data_dir(uuid, 'animate', 'densepose')
    gen_videos = glob(os.path.join(save_dir, '*.mp4'), recursive=True)
    
    return gen_videos

def update_output_video_result(uuid):
    video_list = get_previous_video_result(uuid)
    return video_result_list


def launch_pipeline_densepose(uuid, source_video):
    # check if source_video is end with .mp4
    if not source_video or not source_video.endswith('.mp4'):
        raise gr.Error('请提供一段mp4视频(Please provide 1 mp4 video)')

    before_queue_size = 0
    before_done_count = inference_done_count

    user_directory = os.path.expanduser("~")
    if not os.path.exists(os.path.join(user_directory, '.cache', 'modelscope', 'hub', 'eavesy', 'vid2densepose')):
        gr.Info("第一次初始化会比较耗时，请耐心等待(The first time initialization will take time, please wait)")

    gen_video = DensePose(uuid)

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(gen_video, source_video)

        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = inference_done_count
                to_wait = before_queue_size - (cur_done_count - before_done_count)
                yield ["排队等待资源中，前方还有{}个生成任务(Queueing, there are {} tasks ahead)".format(to_wait, to_wait),
                       None]
            else:
                yield ["生成中, 请耐心等待(Generating, please wait)...", None]
            time.sleep(1)

    output = future.result()
    print(f'生成文件位于路径：{output}')
        
    yield ["生成完毕(Generation done)！", output]




def launch_pipeline_animate(uuid, source_image, motion_sequence, random_seed, sampling_steps, guidance_scale=7.5):
    
    before_queue_size = 0
    before_done_count = inference_done_count

    if not source_image:
        raise gr.Error('请选择一张源图片(Please select 1 source image)')
    
    if not motion_sequence or not motion_sequence.endswith('.mp4'):
        raise gr.Error('请提供一段mp4视频(Please provide 1 mp4 video)')
    

    def read_image(image, size=512):
        return np.array(Image.open(image).resize((size, size)))
    def read_video(video):
        reader = imageio.get_reader(video)
        fps = reader.get_meta_data()['fps']
        return video
    source_image = read_image(source_image)
    motion_sequence = read_video(motion_sequence)

    user_directory = os.path.expanduser("~")
    if not os.path.exists(os.path.join(user_directory, '.cache', 'modelscope', 'hub', 'AI-ModelScope', 'MagicAnimate')):
        gr.Info("第一次初始化会比较耗时，请耐心等待(The first time initialization will take time, please wait)")

    gen_video = MagicAnimate(uuid)

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(gen_video, source_image, motion_sequence, random_seed, sampling_steps, guidance_scale)

        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = inference_done_count
                to_wait = before_queue_size - (cur_done_count - before_done_count)
                yield ["排队等待资源中，前方还有{}个生成任务(Queueing, there are {} tasks ahead)".format(to_wait, to_wait),
                       None]
            else:
                yield ["生成中, 请耐心等待(Generating, please wait)...", None]
            time.sleep(1)

    output = future.result()
    print(f'生成文件位于路径：{output}')
        
    yield ["生成完毕(Generation done)！", output]


def inference_animate():
    def identity_function(inp):
        return inp
    
    with gr.Blocks() as demo:
        uuid = gr.Text(label="modelscope_uuid", visible=False)
        video_result_list = get_previous_video_result(uuid.value)
        print(video_result_list)
        state_video_list = gr.State(value=video_result_list)
        
        gr.Markdown("""该标签页的功能基于[MagicAnimate](https://showlab.github.io/magicanimate/)实现，要使用该标签页，请按照[教程](https://github.com/modelscope/facechain/tree/main/facechain_animate/resources/MagicAnimate/installation_for_magic_animate_ZH.md)安装相关依赖。\n
                    The function of this tab is implemented based on [MagicAnimate](https://showlab.github.io/magicanimate/), to use this tab, you should follow the installation [guide](https://github.com/modelscope/facechain/tree/main/facechain_animate/resources/MagicAnimate/installation_for_magic_animate.md) """)
        
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Box():
                    source_image  = gr.Image(label="源图片(source image)", source="upload", type="filepath")
                    with gr.Column():
                        examples_image=[
                            ["facechain_animate/resources/MagicAnimate/source_image/demo4.png"],
                            ["facechain_animate/resources/MagicAnimate/source_image/0002.png"],
                            ["facechain_animate/resources/MagicAnimate/source_image/dalle8.jpeg"],
                        ]
                        gr.Examples(examples=examples_image, inputs=[source_image],
                                    outputs=[source_image],  fn=identity_function, cache_examples=os.getenv('SYSTEM') == 'spaces', label='Image Example')
                    motion_sequence  = gr.Video(format="mp4", label="动作序列视频(Motion Sequence)", source="upload", height=400)
                    with gr.Column():
                        examples_video=[
                            ["facechain_animate/resources/MagicAnimate/driving/densepose/running.mp4"],
                            ["facechain_animate/resources/MagicAnimate/driving/densepose/demo4.mp4"],
                            ["facechain_animate/resources/MagicAnimate/driving/densepose/running2.mp4"],
                            ["facechain_animate/resources/MagicAnimate/driving/densepose/dancing2.mp4"],
                        ]
                        gr.Examples(examples=examples_video, inputs=[motion_sequence],
                                    outputs=[motion_sequence],  fn=identity_function, cache_examples=os.getenv('SYSTEM') == 'spaces', label='Video Example')
                with gr.Box():
                    gr.Markdown("""
                    注意: 
                    - 如果没有动作序列视频，可以提供原视频文件进行动作序列视频生成（If you don't have motion sequence, you may generate motion sequence from a source video.)
                    - 动作序列视频生成基于DensePose实现（Motion sequence generation is based on DensePose.）
                    """)
                    source_video  = gr.Video(label="原始视频(Original Video)", format="mp4", width=256)

                    gen_motion              = gr.Button("生成动作序列视频(Generate motion sequence)", variant='primary')
                    
                    gen_progress = gr.Textbox(value="当前无生成动作序列视频任务(No motion sequence generation task currently)")
                    
                    gen_motion.click(fn=launch_pipeline_densepose, inputs=[uuid, source_video], 
                        outputs=[gen_progress, motion_sequence])

            with gr.Column(variant='panel'): 
                with gr.Box():
                    gr.Markdown("设置(Settings)")
                    with gr.Column(variant='panel'):
                        random_seed         = gr.Textbox(label="随机种子(Random seed)", value=1, info="default: -1")
                        sampling_steps      = gr.Textbox(label="采样步数(Sampling steps)", value=25, info="default: 25")
                        submit              = gr.Button("生成(Generate)", variant='primary')
                with gr.Box():
                        infer_progress = gr.Textbox(value="当前无任务(No task currently)")
                        gen_video = gr.Video(label="Generated video", format="mp4")

        submit.click(fn=launch_pipeline_animate, inputs=[uuid, source_image, motion_sequence, random_seed, sampling_steps], 
                    outputs=[infer_progress, gen_video])

    return demo


with gr.Blocks(css='style.css') as demo:
    from importlib.util import find_spec
    if find_spec('webui'):
        # if running as a webui extension, don't display banner self-advertisement
        gr.Markdown("# <center> \N{fire} FaceChain Potrait Generation (\N{whale} [Paper cite it here](https://arxiv.org/abs/2308.14256) \N{whale})</center>")
    else:
        gr.Markdown("# <center> \N{fire} FaceChain Potrait Generation ([Github star it here](https://github.com/modelscope/facechain/tree/main) \N{whale},   [Paper](https://arxiv.org/abs/2308.14256) \N{whale},   [API](https://help.aliyun.com/zh/dashscope/developer-reference/facechain-quick-start) \N{whale},   [API's Example App](https://tongyi.aliyun.com/wanxiang/app/portrait-gallery) \N{whale})</center>")
    gr.Markdown("##### <center> 本项目仅供学习交流，请勿将模型及其制作内容用于非法活动或违反他人隐私的场景。(This project is intended solely for the purpose of technological discussion, and should not be used for illegal activities and violating privacy of individuals.)</center>")
    with gr.Tabs():
        with gr.TabItem('\N{clapper board}人物动画生成(Human animate)'):
            inference_animate()

if __name__ == "__main__":
    set_spawn_method()
    demo.queue(status_update_rate=1).launch(share=True)
