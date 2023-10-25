import launch
import os

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("slugify"):
    print("--installing slugify...")
    launch.run_pip("install slugify", "requirements for slugify")
    launch.run_pip("install python-slugify==8.0.1", "requirements for python-slugify")

if not launch.is_installed("modelscope"):
    print("--installing modelscope...")
    launch.run_pip("install modelscope", "requirements for modelscope")

if not launch.is_installed("controlnet_aux"):
    print("--installing controlnet_aux...")
    launch.run_pip("install controlnet_aux==0.0.6", "requirements for controlnet_aux")

if not launch.is_installed("onnxruntime"):
    print("--installing onnxruntime...")
    launch.run_pip("install onnxruntime==1.15.1", "requirements for onnxruntime")

if not launch.is_installed("mmcv"):
    print("--installing mmcv...")
    # Todo 这里有坑
    launch.run_pip("install mmcv-full==1.7.0", "requirements for mmcv")

if not launch.is_installed("mmdet"):
    print("--installing mmdet...")
    launch.run_pip("install mmdet==2.26.0", "requirements for mmdet")

if not launch.is_installed("mediapipe"):
    print("--installing mmdet...")
    launch.run_pip("install mediapipe==0.10.3", "requirements for mediapipe")

if not launch.is_installed("edge_tts"):
    print("--installing edge_tts...")
    launch.run_pip("install edge_tts", "requirements for mediapipe")