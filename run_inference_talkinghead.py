import cv2
import os
from facechain.inference_talkinghead import SadTalker, text_to_speech_edge
from facechain.constants import tts_speakers_map
import asyncio

try:
    import edge_tts
except ImportError:
    print("警告：未找到edge_tts模块，语音合成功能将无法使用。您可以通过`pip install edge-tts`安装它。\n Warning: The edge_tts module is not found, so the speech synthesis function will not be available. You can install it by 'pip install edge-tts'.")

async def text_to_speech_edge(text, speaker, OUTPUT_FILE):
    voice = tts_speakers_map[speaker]
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(OUTPUT_FILE)


save_dir = '.'
source_image = 'lora_result.png'
# source_image = 'tmp_inpaint_left_0.png'

audio_source ='TTS'
audio_tts = None
audio_microphone = None
audio_upload = None
driven_audio = audio_tts

### "语音合成(TTS)", "麦克风(microphone)", "上传文件(upload)"]
if audio_source == "TTS":
    input_text = "欢迎来到英雄联盟!"
    speaker = "普通话(中国大陆)-Xiaoxiao-女"
    OUTPUT_FILE = "tts.mp3"
    asyncio.run(text_to_speech_edge(input_text, speaker, OUTPUT_FILE))
    driven_audio = OUTPUT_FILE
if audio_source == "upload":
    driven_audio = audio_upload
if audio_source == "microphone":
    driven_audio = audio_microphone

### ['crop', 'resize','full']
preprocess = 'crop'
still_mode = True
use_enhancer = False
batch_size = 1
size = 256
pose_style = 0
exp_scale = 1.0

gen_video = SadTalker(save_dir)
video_path = gen_video(source_image, driven_audio, preprocess,
                            still_mode, use_enhancer, batch_size, size, pose_style, exp_scale)
print(video_path)